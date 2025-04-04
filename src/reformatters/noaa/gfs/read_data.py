import hashlib
import os
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import pandas as pd
import rasterio  # type: ignore
import xarray as xr

from reformatters.common.config import Config
from reformatters.common.download import download_to_disk, http_store
from reformatters.common.types import Array2D
from reformatters.noaa.noaa_config_models import NOAADataVar
from reformatters.noaa.noaa_utils import has_hour_0_values

DOWNLOAD_DIR = Path("data/download/")

# Accumulations are reset every 6 hours
GFS_ACCUMULATION_RESET_FREQUENCY_HOURS = 6
GFS_ACCUMULATION_RESET_FREQUENCY_TIMEDELTA: Final[pd.Timedelta] = pd.Timedelta(
    f"{GFS_ACCUMULATION_RESET_FREQUENCY_HOURS}h"
)


class SourceFileCoords(TypedDict):
    init_time: pd.Timestamp
    lead_time: pd.Timedelta


def download_file(
    coords: SourceFileCoords,
    gfs_idx_data_vars: Iterable[NOAADataVar],
) -> tuple[SourceFileCoords, Path | None]:
    init_time = coords["init_time"]
    lead_time = coords["lead_time"]

    lead_time_hours = lead_time.total_seconds() / (60 * 60)
    if lead_time_hours != round(lead_time_hours):
        raise ValueError(f"Lead time {lead_time} must be a whole number of hours")

    init_date_str = init_time.strftime("%Y%m%d")
    init_hour_str = init_time.strftime("%H")

    # Accumulated and last N hour avg values don't exist in the 0-hour forecast.
    if lead_time_hours == 0:
        gfs_idx_data_vars = [
            data_var for data_var in gfs_idx_data_vars if has_hour_0_values(data_var)
        ]

    base_path = f"gfs.{init_date_str}/{init_hour_str}/atmos/gfs.t{init_hour_str}z.pgrb2.0p25.f{lead_time_hours:03.0f}"

    store = http_store("https://noaa-gfs-bdp-pds.s3.amazonaws.com")

    idx_remote_path = f"{base_path}.idx"
    idx_local_path = Path(DOWNLOAD_DIR, f"{base_path}.idx")

    # Create a unique, human debuggable suffix representing the data vars stored in the output file
    vars_str = "-".join(
        var_info.internal_attrs.grib_element for var_info in gfs_idx_data_vars
    )
    vars_hash = digest(
        f"{var_info.internal_attrs.grib_element}:{var_info.internal_attrs.grib_index_level}"
        for var_info in gfs_idx_data_vars
    )
    vars_suffix = f"{vars_str}-{vars_hash}"
    local_path = Path(DOWNLOAD_DIR, f"{base_path}.{vars_suffix}")

    try:
        download_to_disk(
            store,
            idx_remote_path,
            idx_local_path,
            overwrite_existing=not Config.is_dev,
        )

        byte_range_starts, byte_range_ends = parse_index_byte_ranges(
            idx_local_path, gfs_idx_data_vars, int(lead_time_hours)
        )

        download_to_disk(
            store,
            base_path,
            local_path,
            overwrite_existing=not Config.is_dev,
            byte_ranges=(byte_range_starts, byte_range_ends),
        )

        return coords, local_path

    except Exception as e:
        print("Download failed", vars_str, e)
        return coords, None


def digest(data: str | Iterable[str], length: int = 8) -> str:
    """Consistent, likely collision free string digest of one or more strings."""
    if isinstance(data, str):
        data = [data]
    message = hashlib.sha256()
    for string in data:
        message.update(string.encode())
    return message.hexdigest()[:length]


def parse_index_byte_ranges(
    idx_local_path: Path,
    gfs_idx_data_vars: Iterable[NOAADataVar],
    lead_time_hours: int,
) -> tuple[list[int], list[int]]:
    with open(idx_local_path) as index_file:
        index_contents = index_file.read()

    byte_range_starts = []
    byte_range_ends = []

    for var_info in gfs_idx_data_vars:
        if lead_time_hours == 0:
            hours_str_prefix = ""
        elif var_info.attrs.step_type == "instant":
            hours_str_prefix = str(lead_time_hours)
        else:
            diff_hours = lead_time_hours % GFS_ACCUMULATION_RESET_FREQUENCY_HOURS
            if diff_hours == 0:
                reset_hour = lead_time_hours - GFS_ACCUMULATION_RESET_FREQUENCY_HOURS
            else:
                reset_hour = lead_time_hours - diff_hours
            hours_str_prefix = f"{reset_hour}-{lead_time_hours}"

        var_match_str = re.escape(
            f"{var_info.internal_attrs.grib_element}:{var_info.internal_attrs.grib_index_level}:{hours_str_prefix}"
        )
        matches = re.findall(
            f"\\d+:(\\d+):.+:{var_match_str}.+:(\\n\\d+:(\\d+))?",
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


def read_into(
    out: xr.DataArray,
    coords: SourceFileCoords,
    path: os.PathLike[str] | None,
    data_var: NOAADataVar,
) -> None:
    if path is None:
        return  # in rare case file is missing there's nothing to do

    grib_element = data_var.internal_attrs.grib_element
    if data_var.internal_attrs.include_lead_time_suffix:
        lead_hours = coords["lead_time"].total_seconds() / (60 * 60)
        lead_hours_accum = int(lead_hours % GFS_ACCUMULATION_RESET_FREQUENCY_HOURS)
        if lead_hours_accum == 0:
            lead_hours_accum = 6
        grib_element += str(lead_hours_accum).zfill(2)

    try:
        raw_data = read_rasterio(
            path,
            grib_element,
            data_var.internal_attrs.grib_description,
            out.rio.shape,
            out.rio.transform(),
            out.rio.crs,
        )
    except Exception as e:
        print("Read failed", coords, e)
        return

    out.loc[coords] = raw_data


def read_rasterio(
    path: os.PathLike[str],
    grib_element: str,
    grib_description: str,
    out_spatial_shape: tuple[int, int],
    out_transform: rasterio.transform.Affine,
    out_crs: rasterio.crs.CRS,
) -> Array2D[np.float32]:
    with (
        warnings.catch_warnings(),
        rasterio.open(path) as reader,
    ):
        matching_bands = [
            rasterio_band_i
            for band_i in range(reader.count)
            if reader.descriptions[band_i] == grib_description
            and reader.tags(rasterio_band_i := band_i + 1)["GRIB_ELEMENT"]
            == grib_element
        ]

        assert len(matching_bands) == 1, f"Expected exactly 1 matching band, found {matching_bands}. {grib_element=}, {grib_description=}, {path=}"  # fmt: skip
        rasterio_band_index = matching_bands[0]

        assert reader.shape == out_spatial_shape
        assert reader.transform == out_transform
        assert reader.crs.to_dict() == out_crs

        result: Array2D[np.float32] = reader.read(
            rasterio_band_index,
            out_dtype=np.float32,
        )
        return result
