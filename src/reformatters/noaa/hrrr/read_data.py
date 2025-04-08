import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import pandas as pd

from reformatters.common.config import Config
from reformatters.common.download import download_to_disk, http_store
from reformatters.noaa.hrrr.hrrr_config_models import (
    HRRRDataVar,
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
    idx_data_vars: Iterable[HRRRDataVar],
) -> tuple[SourceFileCoords, Path | None]:
    init_time = coords["init_time"]
    lead_time = coords["lead_time"]
    domain = coords["domain"]

    lead_time_hours = lead_time.total_seconds() / (60 * 60)
    if lead_time_hours != round(lead_time_hours):
        raise ValueError(f"Lead time {lead_time} must be a whole number of hours")

    init_date_str = init_time.strftime("%Y%m%d")
    init_hour_str = init_time.strftime("%H")

    remote_path = f"hrrr.{init_date_str}/{domain}/hrrr.t{init_hour_str}z.wrf{hrrr_file_type}f{int(lead_time_hours):02d}.grib2"

    store = http_store("https://noaa-hrrr-bdp-pds.s3.amazonaws.com")

    local_path_filename = remote_path.replace("/", "_")

    idx_remote_path = f"{remote_path}.idx"
    idx_local_path = Path(DOWNLOAD_DIR, f"{local_path_filename}.idx")
    local_path = Path(DOWNLOAD_DIR, local_path_filename)

    try:
        download_to_disk(
            store,
            idx_remote_path,
            idx_local_path,
            overwrite_existing=not Config.is_dev,
        )

        byte_range_starts, byte_range_ends = parse_index_byte_ranges(
            idx_local_path, idx_data_vars
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
        print("Download failed", e)
        return coords, None


def format_hrrr_idx_var(var_info: HRRRDataVar) -> str:
    return f"{var_info.internal_attrs.grib_element}:{var_info.internal_attrs.grib_index_level}"


def parse_index_byte_ranges(
    idx_local_path: Path,
    gefs_idx_data_vars: Iterable[HRRRDataVar],
) -> tuple[list[int], list[int]]:
    with open(idx_local_path) as index_file:
        index_contents = index_file.read()
    byte_range_starts = []
    byte_range_ends = []
    for var_info in gefs_idx_data_vars:
        var_match_str = re.escape(format_hrrr_idx_var(var_info))
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
