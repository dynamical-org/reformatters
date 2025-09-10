import os
import re
import warnings
from typing import Any, Literal, assert_never

import numpy as np
import pandas as pd
import rasterio  # type: ignore
import xarray as xr

from reformatters.common.config import Config
from reformatters.common.logging import get_logger
from reformatters.common.types import Array1D, Array2D, ArrayFloat32
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_ACCUMULATION_RESET_HOURS,
    GEFS_CURRENT_ARCHIVE_START,
    GEFS_REFORECAST_END,
    GEFS_REFORECAST_GRIB_ELEMENT_RENAME,
    GEFSDataVar,
    GefsSourceFileCoord,
)

log = get_logger(__name__)


def is_v12(init_time: pd.Timestamp) -> bool:
    return init_time < GEFS_REFORECAST_END or GEFS_CURRENT_ARCHIVE_START <= init_time


def is_v12_index(times: pd.DatetimeIndex) -> np.ndarray[Any, np.dtype[np.bool]]:
    return (times < GEFS_REFORECAST_END) | (GEFS_CURRENT_ARCHIVE_START <= times)


def is_available_time(times: pd.DatetimeIndex) -> Array1D[np.bool]:
    """Before v12, GEFS files had a 6 hour step."""
    # pre-v12 data is all we have for the 9 month period after the v12 reforecast ends
    # 2019-12-31 and before the v12 forecast archive starts 2020-10-01.
    return is_v12_index(times) | (times.hour % 6 == 0)


def filter_available_times(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return times[is_available_time(times)]


def get_hours_str(var_info: GEFSDataVar, lead_time_hours: float) -> str:
    if lead_time_hours == 0:
        hours_str = "anl"  # analysis
    elif var_info.attrs.step_type == "instant":
        hours_str = f"{lead_time_hours:.0f} hour"
    else:
        diff_hours = lead_time_hours % GEFS_ACCUMULATION_RESET_HOURS
        if diff_hours == 0:
            reset_hour = lead_time_hours - GEFS_ACCUMULATION_RESET_HOURS
        else:
            reset_hour = lead_time_hours - diff_hours
        hours_str = f"{reset_hour:.0f}-{lead_time_hours:.0f} hour"
    return hours_str


def get_grib_element(var_info: GEFSDataVar, init_time: pd.Timestamp) -> str:
    grib_element = var_info.internal_attrs.grib_element
    if init_time < GEFS_REFORECAST_END:
        return GEFS_REFORECAST_GRIB_ELEMENT_RENAME.get(grib_element, grib_element)
    else:
        return grib_element


def format_gefs_idx_var(
    var_info: GEFSDataVar, init_time: pd.Timestamp, lead_time_hours: float
) -> str:
    hours_str = get_hours_str(var_info, lead_time_hours)
    grib_element = get_grib_element(var_info, init_time)
    return f"{grib_element}:{var_info.internal_attrs.grib_index_level}:{hours_str}"


def read_data(
    template: xr.Dataset,
    coord: GefsSourceFileCoord,
    data_var: GEFSDataVar,
) -> ArrayFloat32:
    grib_element = get_grib_element(data_var, coord.init_time)
    if data_var.internal_attrs.include_lead_time_suffix:
        lead_hours = coord.lead_time.total_seconds() / (60 * 60)
        if lead_hours % GEFS_ACCUMULATION_RESET_HOURS == 0:
            grib_element += "06"
        elif lead_hours % GEFS_ACCUMULATION_RESET_HOURS == 3:
            grib_element += "03"
        else:
            raise AssertionError(f"Unexpected lead time hours: {lead_hours}")

    assert coord.downloaded_path is not None
    return read_rasterio(
        coord.downloaded_path,
        grib_element,
        data_var.internal_attrs.grib_description,
        template.rio.shape,
        template.rio.transform(),
        template.rio.crs,
        coord,
        coord.gefs_file_type,
    )


def read_rasterio(
    path: os.PathLike[str],
    grib_element: str,
    grib_description: str,
    out_spatial_shape: tuple[int, int],
    out_transform: rasterio.transform.Affine,
    out_crs: rasterio.crs.CRS,
    coord: GefsSourceFileCoord,
    true_gefs_file_type: Literal["a", "b", "s", "reforecast"],
) -> Array2D[np.float32]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=rasterio.errors.NotGeoreferencedWarning
        )
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

            result: Array2D[np.float32]
            match true_gefs_file_type:
                case "a" | "b":
                    # Interpolate 1.0/0.5 degree data to the 0.25 degree grid.
                    # Every 2nd (0.5 deg) or every 4th (1.0 deg) 0.25 degree pixel's center aligns exactly
                    # with a 1.0/0.5 degree pixel's center.
                    # We use bilinear resampling to retain the exact values from the 1.0/0.5 degree data
                    # at pixels that align, and give a conservative interpolated value for 0.25 degree pixels
                    # that fall between the 1.0/0.5 degree pixels.
                    # Diagram: https://github.com/dynamical-org/reformatters/pull/44#issuecomment-2683799073
                    # Note: having the .read() call interpolate gives very slightly shifted results
                    # so we pay for an extra memory allocation and use reproject to do the interpolation instead.
                    raw = reader.read(rasterio_band_index, out_dtype=np.float32)
                    result, _ = rasterio.warp.reproject(
                        raw,
                        np.full(out_spatial_shape, np.nan, dtype=np.float32),
                        src_transform=reader.transform,
                        src_crs=reader.crs,
                        dst_transform=out_transform,
                        dst_crs=out_crs,
                        resampling=rasterio.warp.Resampling.bilinear,
                    )
                    if not Config.is_prod:
                        # Because the pixel centers are aligned we exactly retain the source data
                        step = 2 if is_v12(coord.init_time) else 4
                        assert np.array_equal(raw, result[::step, ::step])
                    return result
                case "s" | "reforecast":
                    # Confirm the arguments we use to resample 1.0/0.5 degree data
                    # to 0.25 degree grid above match the source 0.25 degree data.
                    assert reader.shape == out_spatial_shape
                    assert reader.transform == out_transform
                    assert reader.crs.to_dict() == out_crs

                    result = reader.read(
                        rasterio_band_index,
                        out_dtype=np.float32,
                    )
                    return result
                case _ as unreachable:
                    assert_never(unreachable)


def parse_grib_index(
    index_contents: str,
    coord: GefsSourceFileCoord,
) -> tuple[list[int], list[int]]:
    byte_range_starts = []
    byte_range_ends = []
    for var_info in coord.data_vars:
        var_match_str = re.escape(
            format_gefs_idx_var(
                var_info, coord.init_time, coord.lead_time.total_seconds() / 3600
            )
        )
        regex = f"\\d+:(\\d+):.+:{var_match_str}.+(\\n\\d+:(\\d+))?"
        matches = re.findall(regex, index_contents)
        assert len(matches) == 1, (
            f"Expected exactly 1 match, found {len(matches)}, {var_info.name} `{regex}` {len(index_contents)}"
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
