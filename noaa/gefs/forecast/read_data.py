import functools
import re
from pathlib import Path
from typing import Any, Literal

import cfgrib  # type: ignore
import pandas as pd
import requests
import xarray as xr
from more_itertools import chunked

from common.config import Config

_STATISTIC_LONG_NAME = {"avg": "ensemble mean", "spr": "ensemble spread"}


def download_file(
    init_time: pd.Timestamp,
    ensemble_member: str | int,
    noaa_file_kind: Literal["a", "b", "s+a", "s+b"],
    lead_time: pd.Timedelta,
    noaa_idx_data_vars: list[dict[str, str]],
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
        ensemble_member = f"{prefix}{ensemble_member:02}"

    if noaa_file_kind == "s+a":
        if lead_time_hours <= 240:
            true_noaa_file_kind = "s"
        else:
            true_noaa_file_kind = "a"
    elif noaa_file_kind == "s+b":
        if lead_time_hours <= 240:
            true_noaa_file_kind = "s"
        else:
            true_noaa_file_kind = "b"
    else:
        true_noaa_file_kind = noaa_file_kind

    file_path = (
        f"gefs.{init_date_str}/{init_hour_str}/atmos/pgrb2sp25/"
        f"ge{ensemble_member}.t{init_hour_str}z.pgrb2{true_noaa_file_kind}.0p25.f{lead_time_hours:03.0f}"
    )
    url = f"https://storage.googleapis.com/gfs-ensemble-forecast-system/{file_path}"

    # Eccodes index files break unless the process working directory is the same as
    # where the files are. To process many files in parallel, we need them all
    # to be in the same directory so we replace / -> - in the file names to put
    # all files in `directory`.
    local_path = Path(directory, file_path.replace("/", "-"))

    idx_url = f"{url}.idx"
    idx_local_path = Path(f"{local_path}.idx")
    download_to_disk(idx_url, idx_local_path, overwrite_existing=not Config.is_dev())

    # TODO pickup here
    byte_range_starts, byte_range_ends = parse_index_byte_ranges(
        idx_local_path, noaa_idx_data_vars
    )
    chunk_starts = chunked(byte_range_starts, 3)
    chunk_ends = chunked(byte_range_ends, 3)
    print(chunk_starts, chunk_ends)

    # TODO use obstore get_ranges_async to download the portions of each file.

    download_to_disk(url, local_path, overwrite_existing=not Config.is_dev())

    return local_path


def parse_index_byte_ranges(
    idx_local_path: Path, noaa_idx_data_vars: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    with open(idx_local_path) as index_file:
        index_contents = index_file.read()
    byte_range_starts = []
    byte_range_ends = []
    for var_info in noaa_idx_data_vars:
        noaa_variable = var_info["noaa_variable"]
        noaa_level = var_info["noaa_level"]
        matches = re.findall(
            f"\\d+:(\\d+):.+:{noaa_variable}:{noaa_level}:.+(\\n\\d+:(\\d+))?",
            index_contents,
        )
        assert len(matches) == 1, f"Expected exactly 1 match, found {matches}"
        match = matches[0]
        start_byte = int(match[0])
        if len(match) == 3:
            end_byte = int(match[2])
        else:
            # TODO run a head request to get the final value,
            # obstore does not support omitting the end byte
            # to go all the way to the end.
            end_byte = -1

        byte_range_starts.append(start_byte)
        byte_range_ends.append(end_byte)
    return byte_range_starts, byte_range_ends


def read_file(file_name: str) -> xr.Dataset:
    """
    Requirement: the process current working directory must contain file_name
    """
    # Open the grib as N different datasets which correctly map grib to xarray datasets
    datasets = cfgrib.open_datasets(file_name)
    datasets = [ds.chunk(-1) for ds in datasets]

    # TODO compat="minimal" is dropping 3 variables
    ds = xr.merge(
        datasets, compat="minimal", join="exact", combine_attrs="no_conflicts"
    )

    renames = {"time": "init_time", "step": "lead_time"}
    ds = ds.expand_dims(tuple(renames.keys())).rename(renames)
    if "number" in ds.coords:
        ds = ds.expand_dims("number", axis=1).rename(number="ensemble_member")
    else:
        # Store summary statistics across ensemble members as separate variables
        # which do not have an ensemble_member dimension. Statistic is either
        # "avg" (ensemble mean) or "spr" (ensemble spread; max minus min)
        statistic = file_name[: file_name.index(".")].removeprefix("ge")
        ds = ds.rename({var: f"{var}_{statistic}" for var in ds.data_vars.keys()})
        for data_var in ds.data_vars.values():
            data_var.attrs["long_name"] = (
                f"{data_var.attrs["long_name"]} ({_STATISTIC_LONG_NAME[statistic]})"
            )

    # Drop level height coords which belong on the variable and are not dataset wide
    ds = ds.drop_vars([c for c in ds.coords if c not in ds.dims])

    # Convert longitude from [0, 360) to [-180, +180)
    ds = ds.assign_coords(longitude=ds["longitude"] - 180)

    return ds


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


def download_to_disk(url: str, local_path: Path, *, overwrite_existing: bool) -> None:
    if not overwrite_existing and local_path.exists():
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    with http_session().get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=None):
                file.write(chunk)
