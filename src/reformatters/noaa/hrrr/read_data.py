import os
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import rasterio  # type: ignore

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
                f"found {len(matches)} in {idx_local_path}: {matches}"
            )

        match = matches[0]
        start_byte = int(match[0])

        if match[2] != "":
            end_byte = int(match[2])
        else:
            # If last idx row, we don't know the length.
            # Add a large offset to get the rest of the file in
            # a way that's compatible with obstore's get_ranges
            end_byte = start_byte + (10 * (2**30))  # +10 GiB

        byte_range_starts.append(start_byte)
        byte_range_ends.append(end_byte)

    return byte_range_starts, byte_range_ends
