import os
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import rasterio  # type: ignore
import xarray as xr

from reformatters.common.config import Config
from reformatters.common.download import download_to_disk, http_store
from reformatters.common.logging import get_logger
from reformatters.common.types import Array2D, ArrayFloat32
from reformatters.noaa.hrrr.hrrr_config_models import (
    HRRRDataVar,
    HRRRDomain,
    HRRRFileType,
)

DOWNLOAD_DIR = Path("data/download/")

logger = get_logger(__name__)


class SourceFileCoords(TypedDict):
    init_time: pd.Timestamp
    lead_time: pd.Timedelta
    domain: HRRRDomain
    file_type: HRRRFileType


# need to refactor based on the file type being associated with each var


def download_file(
    coords: SourceFileCoords,
    idx_data_vars: Iterable[HRRRDataVar],
) -> tuple[SourceFileCoords, Path | None]:
    init_time = coords["init_time"]
    lead_time = coords["lead_time"]
    domain = coords["domain"]
    file_type = coords["file_type"]

    # Verify that the variables in the group are all from the same file type.
    # If not, raise (this should be considered a bug).
    mismatched_file_types = {
        var.internal_attrs.hrrr_file_type
        for var in idx_data_vars
        if var.internal_attrs.hrrr_file_type != file_type
    }
    if mismatched_file_types:
        mismatched_str = ", ".join(ft for ft in mismatched_file_types)
        error_msg = f"All variables must be from {file_type}, but found variables from: {mismatched_str}"
        raise ValueError(error_msg)

    lead_time_hours = lead_time.total_seconds() / (60 * 60)
    if lead_time_hours != round(lead_time_hours):
        raise ValueError(f"Lead time {lead_time} must be a whole number of hours")

    init_date_str = init_time.strftime("%Y%m%d")
    init_hour_str = init_time.strftime("%H")

    remote_path = f"hrrr.{init_date_str}/{domain}/hrrr.t{init_hour_str}z.wrf{file_type}f{int(lead_time_hours):02d}.grib2"

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
        logger.debug(
            "reading byte ranges: %s",
            list(zip(byte_range_starts, byte_range_ends, strict=True)),
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


def read_into(
    out: xr.DataArray,
    coords: SourceFileCoords,
    path: os.PathLike[str] | None,
    data_var: HRRRDataVar,
) -> None:
    if path is None:
        return  # in rare case file is missing there's nothing to do

    grib_element = data_var.internal_attrs.grib_element

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
        logger.exception("Read failed: %s", e)
        return

    # Source file coords contain the domain and file_type, which are not coords in the data array,
    # so we drop them from the coords.
    out_coords = dict(coords)
    del out_coords["domain"]
    del out_coords["file_type"]

    # TODO: Why is the shape of the data not the same as the shape of the data array?
    # out.rio.shape and out.shape are swapped. Need to track this down.
    out.loc[out_coords] = raw_data.T


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
            if grib_description in reader.descriptions[band_i]
            and reader.tags(rasterio_band_i := band_i + 1)["GRIB_ELEMENT"]
            == grib_element
        ]

        assert len(matching_bands) == 1, f"Expected exactly 1 matching band, found {matching_bands}. {grib_element=}, {grib_description=}, {path=}"  # fmt: skip
        rasterio_band_index = matching_bands[0]

        assert reader.shape == out_spatial_shape
        assert reader.transform == out_transform, (
            f"transforms not equal: {reader.transform} != {out_transform}"
        )
        assert reader.crs.to_dict() == out_crs.to_dict(), (
            f"crs not equal: {reader.crs.to_dict()} != {out_crs.to_dict()}"
        )

        result: Array2D[np.float32] = reader.read(
            rasterio_band_index,
            out_dtype=np.float32,
        )
        return result


def read_hrrr_data(
    file_path: os.PathLike[str],
    data_var: HRRRDataVar,
) -> ArrayFloat32:
    """
    Read data from a HRRR GRIB file for a specific variable.

    This is a standalone version of the read_data method that can be called
    from other modules without needing a RegionJob instance.

    Args:
        file_path: Path to the downloaded HRRR GRIB file
        data_var: HRRR data variable configuration

    Returns:
        2D float32 array, transposed to (x, y) order

    Raises:
        ValueError: If the file cannot be read or variable not found
    """
    grib_element = data_var.internal_attrs.grib_element
    grib_description = data_var.internal_attrs.grib_description

    try:
        with (
            warnings.catch_warnings(),
            rasterio.open(file_path) as reader,
        ):
            # Find the matching band for this variable
            matching_bands = [
                band_i + 1  # rasterio uses 1-based indexing
                for band_i in range(reader.count)
                if grib_description in reader.descriptions[band_i]
                and reader.tags(band_i + 1)["GRIB_ELEMENT"] == grib_element
            ]

            if len(matching_bands) != 1:
                raise ValueError(
                    f"Expected exactly 1 matching band for {grib_element}, "
                    f"found {len(matching_bands)} in {file_path}"
                )

            rasterio_band_index = matching_bands[0]

            # Read the data
            result: ArrayFloat32 = reader.read(
                rasterio_band_index,
                out_dtype=np.float32,
            )

            # HRRR data comes in as (y, x) but we need to transpose to (x, y)
            # to match our x, y dimension order in the template
            return result.T

    except Exception as e:
        raise ValueError(f"Failed to read data from {file_path}: {e}") from e


def parse_hrrr_index_byte_ranges(
    idx_local_path: Path,
    data_vars: Iterable[HRRRDataVar],
) -> tuple[list[int], list[int]]:
    """Parse GRIB index file to get byte ranges for specific variables."""
    with open(idx_local_path) as index_file:
        index_contents = index_file.read()

    byte_range_starts = []
    byte_range_ends = []

    for var_info in data_vars:
        var_match_str = f"{var_info.internal_attrs.grib_element}:{var_info.internal_attrs.grib_index_level}"
        var_match_str = re.escape(var_match_str)

        matches = re.findall(
            f"\\d+:(\\d+):.+:{var_match_str}:.+(\\n\\d+:(\\d+))?",
            index_contents,
        )

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly 1 match for {var_info.name}, "
                f"found {len(matches)} in {idx_local_path}"
            )

        match = matches[0]
        start_byte = int(match[0])

        if match[2] != "":
            end_byte = int(match[2])
        else:
            # If no end byte specified, add a large offset
            # (similar to existing logic in read_data.py)
            end_byte = start_byte + (10 * (2**30))  # +10 GiB

        byte_range_starts.append(start_byte)
        byte_range_ends.append(end_byte)

    return byte_range_starts, byte_range_ends


def download_hrrr_file(
    init_time: pd.Timestamp,
    lead_time: pd.Timedelta,
    domain: HRRRDomain,
    file_type: HRRRFileType,
    data_vars: Iterable[HRRRDataVar],
) -> Path:
    """
    Download a HRRR file for specific variables and return the local path.

    This is a standalone version of the download_file method that can be called
    from other modules without needing a RegionJob instance.

    Args:
        init_time: Initialization time of the forecast
        lead_time: Lead time of the forecast
        domain: HRRR domain (typically 'conus')
        file_type: HRRR file type ('sfc', 'prs', 'nat', 'subh')
        data_vars: Variables to download (used for byte range optimization)

    Returns:
        Path to the downloaded GRIB file

    Raises:
        ValueError: If variables don't match file type or lead time is invalid
        FileNotFoundError: If download fails
    """
    # Verify that the variables in the group are all from the same file type
    mismatched_file_types = {
        var.internal_attrs.hrrr_file_type
        for var in data_vars
        if var.internal_attrs.hrrr_file_type != file_type
    }
    if mismatched_file_types:
        mismatched_str = ", ".join(ft for ft in mismatched_file_types)
        error_msg = f"All variables must be from {file_type}, but found variables from: {mismatched_str}"
        raise ValueError(error_msg)

    lead_time_hours = lead_time.total_seconds() / (60 * 60)
    if lead_time_hours != round(lead_time_hours):
        raise ValueError(f"Lead time {lead_time} must be a whole number of hours")

    init_date_str = init_time.strftime("%Y%m%d")
    init_hour_str = init_time.strftime("%H")

    remote_path = f"hrrr.{init_date_str}/{domain}/hrrr.t{init_hour_str}z.wrf{file_type}f{int(lead_time_hours):02d}.grib2"

    store = http_store("https://noaa-hrrr-bdp-pds.s3.amazonaws.com")

    local_path_filename = remote_path.replace("/", "_")

    # Ensure download directory exists
    download_dir = Path(DOWNLOAD_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)

    idx_remote_path = f"{remote_path}.idx"
    idx_local_path = download_dir / f"{local_path_filename}.idx"
    local_path = download_dir / local_path_filename

    try:
        # First download the index file
        download_to_disk(
            store,
            idx_remote_path,
            idx_local_path,
            overwrite_existing=not Config.is_dev,  # Cache files during development
        )

        # Parse the index file to get byte ranges for our variables
        byte_range_starts, byte_range_ends = parse_hrrr_index_byte_ranges(
            idx_local_path, data_vars
        )

        logger.debug(
            "reading byte ranges: %s",
            list(zip(byte_range_starts, byte_range_ends, strict=True)),
        )

        # Download the GRIB file with specific byte ranges
        download_to_disk(
            store,
            remote_path,
            local_path,
            overwrite_existing=not Config.is_dev,
            byte_ranges=(byte_range_starts, byte_range_ends),
        )

        return local_path

    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download HRRR file {remote_path}: {e}"
        ) from e
