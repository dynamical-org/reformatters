import contextlib
import functools
import hashlib
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import obstore  # type: ignore
import pandas as pd
import rasterio  # type: ignore
import requests
import xarray as xr

from common.config import Config

from .config_models import DataVar
from .config_models import NoaaFileType as NoaaFileType

_STATISTIC_LONG_NAME = {"avg": "ensemble mean", "spr": "ensemble spread"}
# The level names needed to grab the right bands from the index
# are different from the ones in the grib itself.
IDX_LEVELS_TO_GRIB_LONG_NAMES = {
    "2 m above ground": '2[m] HTGL="Specified height level above ground"',
    "10 m above ground": '10[m] HTGL="Specified height level above ground"',
}

_VARIABLES_PER_CHUNK = 3


class SourceFileCoords(TypedDict):
    init_time: pd.Timestamp
    ensemble_member: int
    lead_time: pd.Timedelta


def download_file(
    init_time: pd.Timestamp,
    ensemble_member: int,
    lead_time: pd.Timedelta,
    noaa_file_type: NoaaFileType,
    noaa_idx_data_vars: Iterable[DataVar],
    directory: Path,
) -> Path:
    lead_time_hours = lead_time.total_seconds() / (60 * 60)
    if lead_time_hours != round(lead_time_hours):
        raise ValueError(f"Lead time {lead_time} must be a whole number of hours")

    init_date_str = init_time.strftime("%Y%m%d")
    init_hour_str = init_time.strftime("%H")
    if not isinstance(ensemble_member, str):
        # control or perterbed ensemble member
        prefix = "c" if ensemble_member == 0 else "p"
        formatted_ensemble_member = f"{prefix}{ensemble_member:02}"

    # TODO: handle the variables that do not exist in the 0 lead time,
    # but exist in later lead times.
    if noaa_file_type == "s+a":
        if lead_time_hours <= 240:
            true_noaa_file_type = "s"
        else:
            true_noaa_file_type = "a"
    elif noaa_file_type == "s+b":
        if lead_time_hours <= 240:
            true_noaa_file_type = "s"
        else:
            true_noaa_file_type = "b"
    else:
        true_noaa_file_type = noaa_file_type

    remote_path = (
        f"gefs.{init_date_str}/{init_hour_str}/atmos/pgrb2sp25/"
        f"ge{formatted_ensemble_member}.t{init_hour_str}z.pgrb2{true_noaa_file_type}.0p25.f{lead_time_hours:03.0f}"
    )

    store = http_store("https://storage.googleapis.com/gfs-ensemble-forecast-system")

    # Eccodes index files break unless the process working directory is the same as
    # where the files are. To process many files in parallel, we need them all
    # to be in the same directory so we replace / -> - in the file names to put
    # all files in `directory`.
    local_base_file_name = remote_path.replace("/", "_")

    idx_remote_path = f"{remote_path}.idx"
    idx_local_path = Path(f"{local_base_file_name}.idx")

    download_to_disk(
        store, idx_remote_path, idx_local_path, overwrite_existing=not Config.is_dev()
    )

    byte_range_starts, byte_range_ends = parse_index_byte_ranges(
        idx_local_path, noaa_idx_data_vars
    )

    # Create a unique, human debuggable suffix representing the data vars stored in the output file
    vars_str = "-".join(
        var_info.internal_attrs.grib_element for var_info in noaa_idx_data_vars
    )
    vars_hash = digest(format_noaa_idx_var(var_info) for var_info in noaa_idx_data_vars)
    vars_suffix = f"{vars_str}-{vars_hash}"
    local_path = Path(directory, f"{local_base_file_name}.{vars_suffix}")

    print("Downloading", noaa_file_type, vars_suffix)

    download_to_disk(
        store,
        remote_path,
        local_path,
        overwrite_existing=not Config.is_dev(),
        byte_ranges=(byte_range_starts, byte_range_ends),
    )

    return local_path


def parse_index_byte_ranges(
    idx_local_path: Path, noaa_idx_data_vars: Iterable[DataVar]
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


# def read_file(file_name: str) -> xr.Dataset:
#     """
#     Requirement: the process current working directory must contain file_name
#     """
#     # Open the grib as N different datasets which correctly map grib to xarray datasets
#     datasets = cfgrib.open_datasets(file_name)
#     datasets = [ds.chunk(-1) for ds in datasets]

#     # TODO compat="minimal" is dropping 3 variables
#     ds = xr.merge(
#         datasets, compat="minimal", join="exact", combine_attrs="no_conflicts"
#     )

#     renames = {"time": "init_time", "step": "lead_time"}
#     ds = ds.expand_dims(tuple(renames.keys())).rename(renames)
#     if "number" in ds.coords:
#         ds = ds.expand_dims("number", axis=1).rename(number="ensemble_member")
#     else:
#         # Store summary statistics across ensemble members as separate variables
#         # which do not have an ensemble_member dimension. Statistic is either
#         # "avg" (ensemble mean) or "spr" (ensemble spread; max minus min)
#         statistic = file_name[: file_name.index(".")].removeprefix("ge")
#         ds = ds.rename({var: f"{var}_{statistic}" for var in ds.data_vars.keys()})
#         for data_var in ds.data_vars.values():
#             data_var.attrs["long_name"] = (
#                 f"{data_var.attrs["long_name"]} ({_STATISTIC_LONG_NAME[statistic]})"
#             )

#     # Drop level height coords which belong on the variable and are not dataset wide
#     ds = ds.drop_vars([c for c in ds.coords if c not in ds.dims])

#     # Convert longitude from [0, 360) to [-180, +180)
#     ds = ds.assign_coords(longitude=ds["longitude"] - 180)

#     return ds


def read_into(
    out: xr.DataArray,
    coords: SourceFileCoords,
    path: os.PathLike[str],
    grib_element: str,
    grib_description: str,
) -> None:
    out.loc[coords] = read_rasterio(path, grib_element, grib_description)


def read_rasterio(
    path: os.PathLike[str], grib_element: str, grib_description: str
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
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
        # client_options={
        #     "connect_timeout": "4 seconds",
        #     "timeout": "16 seconds",
        # },
        # retry_config={
        #     "max_retries": 10,
        #     "backoff": {
        #         "base": 2,
        #         "init_backoff": timedelta(seconds=1),
        #         "max_backoff": timedelta(seconds=16),
        #     },
        #     # A backstop, shouldn't hit this with the above backoff settings
        #     "retry_timeout": timedelta(minutes=3),
        # },
    )


@functools.cache
def http_session() -> requests.Session:
    """
    A requests.Session tuned to maximize chance of success at the expense of latency,
    while not waiting indefinitely for unresponsive servers.

    Uses a backoff retry for 500 level responses and connection errors.
    Applies a default connection and read timeout if one isn't specificed in the request.
    """
    session = requests.Session()
    retry = requests.adapters.Retry(
        total=10,
        redirect=1,
        backoff_factor=0.5,
        backoff_jitter=0.5,
        status_forcelist=(500, 502, 503, 504),
    )
    # default_timeout tuple is (connection timeout, read timeout)
    adapter = DefaultTimeoutHTTPAdapter(default_timeout=(4, 16), max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


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
