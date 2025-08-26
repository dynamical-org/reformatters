import os
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import rasterio  # type: ignore

from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import digest
from reformatters.common.logging import get_logger
from reformatters.common.types import ArrayFloat32
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
            # For HRRR, we need more flexible matching since the template descriptions
            # might not match exactly what's in the GRIB file
            matching_bands = []
            for band_i in range(reader.count):
                band_idx = band_i + 1  # rasterio uses 1-based indexing
                tags = reader.tags(band_idx)

                # Always check GRIB_ELEMENT matches
                if tags.get("GRIB_ELEMENT") != grib_element:
                    continue

                # For matching, we'll be more flexible with descriptions
                # Check if the band description or tags contain indicators of the expected level
                description = reader.descriptions[band_i] if reader.descriptions else ""

                # Try exact description match first (for backward compatibility)
                if grib_description in description:
                    matching_bands.append(band_idx)
                    continue

                # For HRRR, try matching based on grib_index_level patterns
                grib_index_level = data_var.internal_attrs.grib_index_level.lower()

                if "entire atmosphere" in grib_index_level:
                    # Look for EATM (Entire Atmosphere) in description or tags
                    if (
                        "eatm" in description.lower()
                        or "entire atmosphere" in description.lower()
                    ):
                        matching_bands.append(band_idx)
                        continue
                elif "surface" in grib_index_level:
                    # Look for surface indicators
                    if "sfc" in description.lower() or "surface" in description.lower():
                        matching_bands.append(band_idx)
                        continue
                else:
                    # For other levels, try to match the level description
                    if grib_index_level in description.lower():
                        matching_bands.append(band_idx)
                        continue

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
            return result

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
    # Import here to avoid circular imports
    from reformatters.noaa.hrrr.forecast_48_hour.region_job import HRRRSourceFileCoord

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

    # Create a coordinate to use its URL generation logic
    coord = HRRRSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        domain=domain,
        file_type=file_type,
    )

    try:
        # Download index file and parse byte ranges
        idx_url = coord.get_idx_url()
        logger.debug("Downloading HRRR index file from %s", idx_url)
        idx_local_path = http_download_to_disk(idx_url, "hrrr-forecast-48h")

        # Parse the index file to get byte ranges for our variables
        byte_range_starts, byte_range_ends = parse_hrrr_index_byte_ranges(
            idx_local_path, data_vars
        )

        logger.debug(
            "reading byte ranges: %s",
            list(zip(byte_range_starts, byte_range_ends, strict=True)),
        )

        # Create a suffix for the downloaded file based on the byte ranges
        vars_suffix = digest(
            f"{s}-{e}" for s, e in zip(byte_range_starts, byte_range_ends, strict=True)
        )

        # Download the GRIB file with specific byte ranges
        return http_download_to_disk(
            coord.get_url(),
            "hrrr-forecast-48h",
            byte_ranges=(byte_range_starts, byte_range_ends),
            local_path_suffix=f"-{vars_suffix}",
        )

    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download HRRR file {coord.get_url()}: {e}"
        ) from e
