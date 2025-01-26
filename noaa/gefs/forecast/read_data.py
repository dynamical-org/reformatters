import contextlib
import functools
import hashlib
import os
import re
from collections.abc import Iterable, Sequence
from datetime import timedelta
from itertools import product
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import obstore  # type: ignore
import pandas as pd
import rasterio  # type: ignore
import requests
import xarray as xr

from common.config import Config
from common.types import Array2D

from .config_models import DataVar, EnsembleStatistic
from .config_models import NoaaFileType as NoaaFileType

FILE_RESOLUTIONS = {
    "s": "0p25",
    "a": "0p50",
    "b": "0p50",
}


class EnsembleSourceFileCoords(TypedDict):
    init_time: pd.Timestamp
    ensemble_member: int
    lead_time: pd.Timedelta


class StatisticSourceFileCoords(TypedDict):
    init_time: pd.Timestamp
    statistic: EnsembleStatistic
    lead_time: pd.Timedelta


type SourceFileCoords = EnsembleSourceFileCoords | StatisticSourceFileCoords


class ChunkCoordinates(TypedDict):
    ensemble: Iterable[EnsembleSourceFileCoords]
    statistic: Iterable[StatisticSourceFileCoords]


def download_file(
    coords: SourceFileCoords,
    noaa_file_type: NoaaFileType,
    noaa_idx_data_vars: Iterable[DataVar],
    directory: Path,
) -> tuple[SourceFileCoords, Path | None]:
    init_time = coords["init_time"]
    lead_time = coords["lead_time"]

    lead_time_hours = lead_time.total_seconds() / (60 * 60)
    if lead_time_hours != round(lead_time_hours):
        raise ValueError(f"Lead time {lead_time} must be a whole number of hours")

    init_date_str = init_time.strftime("%Y%m%d")
    init_hour_str = init_time.strftime("%H")

    if isinstance(ensemble_member := coords.get("ensemble_member"), int | np.integer):
        # control (c) or perterbed (p) ensemble member
        prefix = "c" if ensemble_member == 0 else "p"
        ensemble_or_statistic_str = f"{prefix}{ensemble_member:02}"
    elif isinstance(statistic := coords.get("statistic"), str):
        ensemble_or_statistic_str = statistic
    else:
        raise ValueError(f"coords must be ensemble or statistic coord, found {coords}.")

    # Accumulated values don't exist in the 0-hour forecast.
    if lead_time_hours == 0:
        noaa_idx_data_vars = [
            data_var
            for data_var in noaa_idx_data_vars
            if data_var.attrs.step_type != "accum"
        ]

    true_noaa_file_type = get_noaa_file_type_for_lead_time(lead_time, noaa_file_type)

    remote_path = (
        f"gefs.{init_date_str}/{init_hour_str}/atmos/pgrb2{true_noaa_file_type}{FILE_RESOLUTIONS[true_noaa_file_type].strip("0")}/"
        f"ge{ensemble_or_statistic_str}.t{init_hour_str}z.pgrb2{true_noaa_file_type}.{FILE_RESOLUTIONS[true_noaa_file_type]}.f{lead_time_hours:03.0f}"
    )

    store = http_store("https://storage.googleapis.com/gfs-ensemble-forecast-system")

    # Eccodes index files break unless the process working directory is the same as
    # where the files are. To process many files in parallel, we need them all
    # to be in the same directory so we replace / -> - in the file names to put
    # all files in `directory`.
    local_base_file_name = remote_path.replace("/", "_")

    idx_remote_path = f"{remote_path}.idx"
    idx_local_path = Path(f"{local_base_file_name}.idx")

    # Create a unique, human debuggable suffix representing the data vars stored in the output file
    vars_str = "-".join(
        var_info.internal_attrs.grib_element for var_info in noaa_idx_data_vars
    )
    vars_hash = digest(format_noaa_idx_var(var_info) for var_info in noaa_idx_data_vars)
    vars_suffix = f"{vars_str}-{vars_hash}"
    local_path = Path(directory, f"{local_base_file_name}.{vars_suffix}")

    try:
        download_to_disk(
            store,
            idx_remote_path,
            idx_local_path,
            overwrite_existing=not Config.is_dev(),
        )

        byte_range_starts, byte_range_ends = parse_index_byte_ranges(
            idx_local_path, noaa_idx_data_vars
        )

        # print("Downloading", remote_path.split("/")[-1], vars_suffix)

        download_to_disk(
            store,
            remote_path,
            local_path,
            overwrite_existing=not Config.is_dev(),
            byte_ranges=(byte_range_starts, byte_range_ends),
        )

        return coords, local_path

    except Exception as e:
        print("Download failed", vars_str, e)
        return coords, None


def get_noaa_file_type_for_lead_time(
    lead_time: pd.Timedelta, noaa_file_type: NoaaFileType
) -> Literal["a", "b", "s"]:
    if noaa_file_type == "s+a":
        if lead_time <= pd.Timedelta(hours=240):
            return "s"
        else:
            return "a"
    elif noaa_file_type == "s+b":
        if lead_time <= pd.Timedelta(hours=240):
            return "s"
        else:
            return "b"
    else:
        return noaa_file_type


def generate_chunk_coordinates(
    chunk_init_times: Iterable[pd.Timestamp],
    chunk_ensemble_members: Iterable[int],
    chunk_lead_times: Iterable[pd.Timedelta],
    statistics: Iterable[EnsembleStatistic],
) -> ChunkCoordinates:
    chunk_coords_ensemble: list[EnsembleSourceFileCoords] = [
        {
            "init_time": init_time,
            "ensemble_member": ensemble_member,
            "lead_time": lead_time,
        }
        for init_time, ensemble_member, lead_time in product(
            chunk_init_times, chunk_ensemble_members, chunk_lead_times
        )
    ]

    chunk_coords_statistic: list[StatisticSourceFileCoords] = [
        {
            "init_time": init_time,
            "statistic": statistic,
            "lead_time": lead_time,
        }
        for init_time, statistic, lead_time in product(
            chunk_init_times, statistics, chunk_lead_times
        )
    ]
    return {
        "ensemble": chunk_coords_ensemble,
        "statistic": chunk_coords_statistic,
    }


def parse_index_byte_ranges(
    idx_local_path: Path,
    noaa_idx_data_vars: Iterable[DataVar],
) -> tuple[list[int], list[int]]:
    with open(idx_local_path) as index_file:
        index_contents = index_file.read()
    byte_range_starts = []
    byte_range_ends = []
    for var_info in noaa_idx_data_vars:
        var_match_str = re.escape(format_noaa_idx_var(var_info))
        matches = re.findall(
            f"\\d+:(\\d+):.+:{var_match_str}:.+(\\n\\d+:(\\d+))?",
            index_contents,
        )
        assert (
            len(matches) == 1
        ), f"Expected exactly 1 match, found {matches}, {var_info=}"
        match = matches[0]
        start_byte = int(match[0])
        if match[2] != "":
            end_byte = int(match[2])
        else:
            # TODO run a head request to get the final value,
            # obstore does not support omitting the end byte
            # to go all the way to the end.
            raise NotImplementedError(
                "special handling needed for last variable in index"
            )

        byte_range_starts.append(start_byte)
        byte_range_ends.append(end_byte)
    return byte_range_starts, byte_range_ends


def format_noaa_idx_var(var_info: DataVar) -> str:
    return f"{var_info.internal_attrs.grib_element}:{var_info.internal_attrs.grib_index_level}"


def digest(data: str | Iterable[str], length: int = 8) -> str:
    """Consistent, likely collision free string digest of one or more strings."""
    if isinstance(data, str):
        data = [data]
    message = hashlib.sha256()
    for string in data:
        message.update(string.encode())
    return message.hexdigest()[:length]


def read_into(
    out: xr.DataArray,
    coords: SourceFileCoords,
    path: os.PathLike[str] | None,
    data_var: DataVar,
) -> None:
    if path is None:
        return  # in rare case file is missing there's nothing to do

    grib_element = data_var.internal_attrs.grib_element
    if data_var.internal_attrs.include_lead_time_suffix:
        lead_hours = coords["lead_time"].total_seconds() / (60 * 60)
        if lead_hours % 6 == 0:
            grib_element += "06"
        elif lead_hours % 6 == 3:
            grib_element += "03"
        else:
            raise AssertionError(f"Unexpected lead time hours: {lead_hours}")

    # The statistic "dimension" is flattened into the variable names (eg. temperature_2m_avg),
    # so remove the statistic coordinate from coords used to index into xarray Dataset/DataArray.
    if "statistic" in coords:
        ds_coords = {k: v for k, v in coords.items() if k != "statistic"}
    else:
        ds_coords = dict(coords)

    try:
        raw_data = read_rasterio(
            path,
            grib_element,
            data_var.internal_attrs.grib_description,
        )
    except Exception as e:
        print("Read failed", coords, e)
        return

    out.loc[ds_coords] = maybe_resample(raw_data, coords, data_var)


def maybe_resample(
    raw_data: Array2D[np.float32],
    coords: SourceFileCoords,
    data_var: DataVar,
) -> Array2D[np.float32]:
    noaa_file_type = get_noaa_file_type_for_lead_time(
        coords["lead_time"], data_var.internal_attrs.noaa_file_type
    )
    if noaa_file_type in ["a", "b"]:
        # Duplicate 1 pixel into 4 to go from 361x720 (a and b file resolution) to 721x1440 (s file resolution)
        # TODO figure out least worst translation from 361 -> 721 pixels
        len_lat, len_lon = raw_data.shape
        lat_ix = np.arange(len_lat).repeat(2)[:-1]
        lon_ix = np.arange(len_lon).repeat(2)
        return raw_data[np.ix_(lat_ix, lon_ix)]  # type: ignore

    return raw_data


def read_rasterio(
    path: os.PathLike[str], grib_element: str, grib_description: str
) -> Array2D[np.float32]:
    with rasterio.open(path) as reader:
        matching_bands = [
            rasterio_band_i
            for band_i in range(reader.count)
            if reader.descriptions[band_i] == grib_description
            and reader.tags(rasterio_band_i := band_i + 1)["GRIB_ELEMENT"]
            == grib_element
        ]

        assert len(matching_bands) == 1, f"Expected exactly 1 matching band, found {matching_bands}. {grib_element=}, {grib_description=}, {path=}"  # fmt: skip
        rasterio_band_index = matching_bands[0]

        return reader.read(rasterio_band_index, out_dtype=np.float32)  # type: ignore


class DefaultTimeoutHTTPAdapter(requests.adapters.HTTPAdapter):
    def __init__(
        self, *args: Any, default_timeout: float | tuple[float, float], **kwargs: Any
    ):
        self.default_timeout = default_timeout
        super().__init__(*args, **kwargs)

    def send(self, *args: Any, **kwargs: Any) -> requests.Response:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.default_timeout
        return super().send(*args, **kwargs)


@functools.cache
def http_store(base_url: str) -> obstore.store.HTTPStore:
    """
    A obstore.store.HTTPStore tuned to maximize chance of success at the expense
    of latency, while not waiting indefinitely for unresponsive servers.
    """
    return obstore.store.HTTPStore.from_url(
        base_url,
        client_options={
            "connect_timeout": "4 seconds",
            "timeout": "16 seconds",
        },
        retry_config={
            "max_retries": 16,
            "backoff": {
                "base": 2,
                "init_backoff": timedelta(seconds=1),
                "max_backoff": timedelta(seconds=16),
            },
            # A backstop, shouldn't hit this with the above backoff settings
            "retry_timeout": timedelta(minutes=5),
        },
    )


def download_to_disk(
    store: obstore.store.HTTPStore,
    path: str,
    local_path: Path,
    *,
    byte_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
    overwrite_existing: bool,
) -> None:
    if not overwrite_existing and local_path.exists():
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if byte_ranges is not None:
        byte_range_starts, byte_range_ends = byte_ranges[0], byte_ranges[1]
        response_buffers = obstore.get_ranges(
            store=store, path=path, starts=byte_range_starts, ends=byte_range_ends
        )
    else:
        response_buffers = obstore.get(store, path).stream()

    try:
        with open(local_path, "wb") as file:
            for buffer in response_buffers:
                file.write(buffer)

    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.remove(local_path)
        raise
