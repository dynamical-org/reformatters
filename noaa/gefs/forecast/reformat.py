import functools
import os
import tempfile

import cfgrib  # type: ignore
import pandas as pd
import requests
import xarray as xr

from common.config import Config, Env


def download_and_load_source_file(
    init_time: pd.Timestamp, lead_time: pd.Timedelta
) -> xr.Dataset:
    lead_time_hours = lead_time.total_seconds() / (60 * 60)
    if lead_time_hours != round(lead_time_hours):
        raise ValueError(f"Lead time {lead_time} must be a whole number of hours")

    init_date_str = init_time.strftime("%Y%m%d")
    init_hour_str = init_time.strftime("%H")

    file_path = (
        f"gefs.{init_date_str}/{init_hour_str}/atmos/pgrb2sp25/"
        f"geavg.t{init_hour_str}z.pgrb2s.0p25.f{lead_time_hours:03.0f}"
    )
    url = f"https://storage.googleapis.com/gfs-ensemble-forecast-system/{file_path}"

    # cfgrib/eccodes appears to only read files from disk, download there first
    with tempfile.NamedTemporaryFile() as file:
        is_dev = Config.env == Env.dev
        local_path = file.name if not is_dev else "/tmp/gefs-cache/" + file_path  # noqa: S108 Only allow reads from static file paths in dev
        download(url, local_path, overwrite_existing=is_dev)

        # Open the grib as N different datasets to pedantically map grib to xarray
        datasets = cfgrib.open_datasets(local_path)

        # TODO compat="minimal" is dropping 3 variables
        ds = xr.merge(
            datasets, compat="minimal", join="exact", combine_attrs="no_conflicts"
        )

        renames = {"time": "init_time", "step": "lead_time"}
        ds = ds.expand_dims(tuple(renames.keys())).rename(renames)

        # TODO move cordinates about level height to attributes on their specific vars
        ds = ds.drop_vars([c for c in ds.coords if c not in ds.dims])

        # Convert longitude from [0, 360) to [-180, +180)
        ds = ds.assign_coords(longitude=ds["longitude"] - 180)

        ds.load()

    return ds


@functools.cache
def http_session():
    session = requests.Session()
    retry = requests.adapters.Retry(
        total=5,
        redirect=1,
        backoff_factor=0.5,
        backoff_jitter=0.5,
        status_forcelist=(500, 502, 503, 504),
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download(url: str, local_path: str, *, overwrite_existing: bool):
    if not overwrite_existing and os.path.isfile(local_path):
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print("Downloading", url)
    with http_session().get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=None):
                file.write(chunk)
