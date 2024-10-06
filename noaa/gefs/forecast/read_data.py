import functools
from pathlib import Path

import cfgrib  # type: ignore
import pandas as pd
import requests
import xarray as xr

from common.config import Config
from noaa.gefs.forecast import template

_STATISTIC_LONG_NAME = {"avg": "ensemble mean", "spr": "ensemble spread"}


def download_file(
    init_time: pd.Timestamp,
    ensemble_member: str | int,
    lead_time: pd.Timedelta,
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

    file_path = (
        f"gefs.{init_date_str}/{init_hour_str}/atmos/pgrb2sp25/"
        f"ge{ensemble_member}.t{init_hour_str}z.pgrb2s.0p25.f{lead_time_hours:03.0f}"
    )
    url = f"https://storage.googleapis.com/gfs-ensemble-forecast-system/{file_path}"
    idx_url = f"{url}.idx"

    local_path = Path(directory, file_path)
    idx_local_path = Path(f"{local_path}.idx")

    download(url, local_path, overwrite_existing=not Config.is_dev())
    download(idx_url, idx_local_path, overwrite_existing=not Config.is_dev())

    return local_path


def read_file(path: Path) -> xr.Dataset:
    # Open the grib as N different datasets which correctly map grib to xarray datasets
    datasets = cfgrib.open_datasets(path)
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
        statistic = path.name[: path.name.index(".")].removeprefix("ge")
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


@functools.cache
def http_session() -> requests.Session:
    session = requests.Session()
    retry = requests.adapters.Retry(
        total=10,
        redirect=1,
        backoff_factor=0.5,
        backoff_jitter=0.5,
        status_forcelist=(500, 502, 503, 504),
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download(url: str, local_path: Path, *, overwrite_existing: bool) -> None:
    if not overwrite_existing and local_path.exists():
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading", url)
    with http_session().get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=None):
                file.write(chunk)
