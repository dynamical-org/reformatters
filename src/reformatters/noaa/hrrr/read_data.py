from pathlib import Path
from typing import TypedDict

import pandas as pd

from reformatters.common.config import Config
from reformatters.common.downloading import download_to_disk, http_store
from reformatters.noaa.hrrr.hrrr_config_models import (
    HRRRDomain,
    HRRRFileType,
)

DOWNLOAD_DIR = Path("data/download/")


class SourceFileCoords(TypedDict):
    init_time: pd.Timestamp
    lead_time: pd.Timedelta
    domain: HRRRDomain


def download_file(
    coords: SourceFileCoords,
    hrrr_file_type: HRRRFileType,
    # TODO add variable support
    # idx_data_vars: Iterable[HRRRDataVar],
) -> tuple[SourceFileCoords, Path | None]:
    init_time = coords["init_time"]
    lead_time = coords["lead_time"]
    domain = coords["domain"]

    lead_time_hours = lead_time.total_seconds() / (60 * 60)
    if lead_time_hours != round(lead_time_hours):
        raise ValueError(f"Lead time {lead_time} must be a whole number of hours")

    init_date_str = init_time.strftime("%Y%m%d")
    init_hour_str = init_time.strftime("%H")

    print(lead_time_hours)

    remote_path = f"hrrr.{init_date_str}/{domain}/hrrr.t{init_hour_str}z.wrf{hrrr_file_type}f{int(lead_time_hours):02d}.grib2"

    store = http_store("https://noaa-hrrr-bdp-pds.s3.amazonaws.com")

    local_path_filename = remote_path.replace("/", "_")

    idx_remote_path = f"{remote_path}.idx"
    idx_local_path = Path(DOWNLOAD_DIR, f"{local_path_filename}.idx")
    local_path = Path(DOWNLOAD_DIR, local_path_filename)

    try:
        download_to_disk(
            store, remote_path, local_path, overwrite_existing=not Config.is_dev
        )
        download_to_disk(
            store, idx_remote_path, idx_local_path, overwrite_existing=not Config.is_dev
        )

        return coords, local_path
    except Exception as e:
        print("Download failed", e)
        return coords, None
