import contextlib
import functools
import hashlib
import os
import re
from collections.abc import Iterable, Sequence
from datetime import timedelta
from itertools import product
from pathlib import Path
from typing import Any, Literal, TypedDict, assert_never

import numpy as np
import obstore
import pandas as pd
import rasterio  # type: ignore
import requests
import xarray as xr

from reformatters.common.config import Config
from reformatters.common.config_models import EnsembleStatistic
from reformatters.common.types import Array2D
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar, GEFSFileType

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
    gefs_file_type: GEFSFileType,
    gefs_idx_data_vars: Iterable[GEFSDataVar],
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

    # Accumulated and last N hour avg values don't exist in the 0-hour forecast.
    if lead_time_hours == 0:
        gefs_idx_data_vars = [
            data_var
            for data_var in gefs_idx_data_vars
            if data_var.attrs.step_type not in ("accum", "avg")
        ]

    true_gefs_file_type = get_gefs_file_type_for_lead_time(lead_time, gefs_file_type)

    remote_path = (
        f"gefs.{init_date_str}/{init_hour_str}/atmos/pgrb2{true_gefs_file_type}{FILE_RESOLUTIONS[true_gefs_file_type].strip('0')}/"
        f"ge{ensemble_or_statistic_str}.t{init_hour_str}z.pgrb2{true_gefs_file_type}.{FILE_RESOLUTIONS[true_gefs_file_type]}.f{lead_time_hours:03.0f}"
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
        var_info.internal_attrs.grib_element for var_info in gefs_idx_data_vars
    )
    vars_hash = digest(format_gefs_idx_var(var_info) for var_info in gefs_idx_data_vars)
    vars_suffix = f"{vars_str}-{vars_hash}"
    local_path = Path(directory, f"{local_base_file_name}.{vars_suffix}")

    try:
        download_to_disk(
            store,
            idx_remote_path,
            idx_local_path,
            overwrite_existing=not Config.is_dev,
        )

        byte_range_starts, byte_range_ends = parse_index_byte_ranges(
            idx_local_path, gefs_idx_data_vars
        )

        download_to_disk(
            store,
            remote_path,
            local_path,
            overwrite_existing=not Config.is_dev,
            byte_ranges=(byte_range_starts, byte_range_ends),
        )

        return coords, local_path

    except Exception as e:
        print("Download failed", vars_str, e)
        return coords, None


def get_gefs_file_type_for_lead_time(
    lead_time: pd.Timedelta, gefs_file_type: GEFSFileType
) -> Literal["a", "b", "s"]:
    if gefs_file_type == "s+a":
        if lead_time <= pd.Timedelta(hours=240):
            return "s"
        else:
            return "a"
    elif gefs_file_type == "s+b":
        if lead_time <= pd.Timedelta(hours=240):
            return "s"
        else:
            return "b"
    else:
        return gefs_file_type


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
    gefs_idx_data_vars: Iterable[GEFSDataVar],
) -> tuple[list[int], list[int]]:
    with open(idx_local_path) as index_file:
        index_contents = index_file.read()
    byte_range_starts = []
    byte_range_ends = []
    for var_info in gefs_idx_data_vars:
        var_match_str = re.escape(format_gefs_idx_var(var_info))
        matches = re.findall(
            f"\\d+:(\\d+):.+:{var_match_str}:.+(\\n\\d+:(\\d+))?",
            index_contents,
        )
        assert len(matches) == 1, (
            f"Expected exactly 1 match, found {matches}, {var_info.name} {idx_local_path}"
        )
        match = matches[0]
        start_byte = int(match[0])
        if match[2] != "":
            end_byte = int(match[2])
        else:
            # obstore does not support omitting the end byte
            # to go all the way to the end of the file, but if
            # you give a value off the end of the file you get
            # the rest of the bytes in the file and no more.
            # So we add a big number, but not too big because
            # it has to fit in a usize in rust object store,
            # and hope this code is never adapted to pull
            # individual grib messages > 10 GiB.
            # GEFS messages are ~0.5MB.
            end_byte = start_byte + (10 * (2**30))  # +10 GiB

        byte_range_starts.append(start_byte)
        byte_range_ends.append(end_byte)
    return byte_range_starts, byte_range_ends


def format_gefs_idx_var(var_info: GEFSDataVar) -> str:
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
    data_var: GEFSDataVar,
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
            out.rio.shape,
            out.rio.transform(),
            out.rio.crs,
            coords,
            data_var.internal_attrs.gefs_file_type,
        )
    except Exception as e:
        print("Read failed", coords, e)
        return

    out.loc[ds_coords] = raw_data


def read_rasterio(
    path: os.PathLike[str],
    grib_element: str,
    grib_description: str,
    out_spatial_shape: tuple[int, int],
    out_transform: rasterio.transform.Affine,
    out_crs: rasterio.crs.CRS,
    coords: SourceFileCoords,
    gefs_file_type: GEFSFileType,
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

        result: Array2D[np.float32]
        match get_gefs_file_type_for_lead_time(coords["lead_time"], gefs_file_type):
            case "a" | "b":
                # Interpolate 0.5 degree data to the 0.25 degree grid.
                # Every other 0.25 degree pixel's center aligns exactly with a 0.5 degree pixel's center.
                # We use bilinear resampling to retain the exact values from the 0.5 degree data
                # at pixels that align, and give a conservative interpolated value for 0.25 degree pixels
                # that fall between the 0.5 degree pixels.
                # Diagram: https://github.com/dynamical-org/reformatters/pull/44#issuecomment-2683799073
                # Note: having the .read() call interpolate gives very slightly shifted results
                # so we pay for an extra memory allocation and use reproject to do the interpolation instead.
                raw = reader.read(rasterio_band_index, out_dtype=np.float32)
                result, _ = rasterio.warp.reproject(
                    raw,
                    np.full(out_spatial_shape, np.nan, dtype=np.float32),
                    src_transform=reader.transform,
                    src_crs=reader.crs,
                    dst_transform=out_transform,
                    dst_crs=out_crs,
                    resampling=rasterio.warp.Resampling.bilinear,
                )
                if not Config.is_prod:
                    # Because the pixel centers are aligned we exactly retain the source data
                    assert np.array_equal(raw, result[::2, ::2])
                return result
            case "s":
                # Confirm the arguments we use to resample 0.5 degree data
                # to 0.25 degree grid above match the source 0.25 degree data.
                assert reader.shape == out_spatial_shape
                assert reader.transform == out_transform
                assert reader.crs.to_dict() == out_crs

                result = reader.read(
                    rasterio_band_index,
                    out_dtype=np.float32,
                )
                return result
            case _ as unreachable:
                assert_never(unreachable)


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

    response_buffers: obstore.BytesStream | list[obstore.Bytes]
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
