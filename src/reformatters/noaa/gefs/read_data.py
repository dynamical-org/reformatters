import os
import warnings
from typing import Literal, assert_never

import numpy as np
import rasterio
import rasterio.warp
import rioxarray  # noqa: F401  # Registers .rio accessor on xarray objects
import xarray as xr

from reformatters.common.config import Config
from reformatters.common.types import Array2D, ArrayFloat32
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_ACCUMULATION_RESET_HOURS,
    GEFSDataVar,
    GefsSourceFileCoord,
    get_grib_element,
    is_v12,
)


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
            matching_bands: list[int] = []
            for band_i in range(reader.count):
                rasterio_band_i = band_i + 1
                if (
                    reader.descriptions[band_i] == grib_description
                    and reader.tags(rasterio_band_i)["GRIB_ELEMENT"] == grib_element
                ):
                    matching_bands.append(rasterio_band_i)

            assert len(matching_bands) == 1, f"Expected exactly 1 matching band, found {matching_bands}. {grib_element=}, {grib_description=}, {path=}"  # fmt: skip
            rasterio_band_index = matching_bands[0]

            result: Array2D[np.float32]
            match true_gefs_file_type:
                case "s":
                    # Confirm the arguments we use to resample 1.0/0.5 degree data
                    # to 0.25 degree grid below match the source 0.25 degree data.
                    assert reader.shape == out_spatial_shape
                    assert reader.transform == out_transform
                    assert reader.crs.to_dict() == out_crs
                    return reader.read(rasterio_band_index, out_dtype=np.float32)
                case "a" | "b" | "reforecast":
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
                    if reader.shape == out_spatial_shape:
                        # Some reforecast files are already 0.25° - no reprojection needed.
                        assert reader.transform == out_transform
                        assert reader.crs.to_dict() == out_crs
                        return raw
                    result = _reproject_bilinear_longitude_wrap(
                        raw,
                        reader.transform,
                        reader.crs,
                        out_spatial_shape,
                        out_transform,
                        out_crs,
                    )
                    if not Config.is_prod:
                        # Because the pixel centers are aligned we exactly retain the source data
                        step = 2 if is_v12(coord.init_time) else 4
                        assert np.array_equal(raw, result[::step, ::step])
                    return result
                case _ as unreachable:
                    assert_never(unreachable)


def _reproject_bilinear_longitude_wrap(
    raw: Array2D[np.float32],
    src_transform: rasterio.transform.Affine,
    src_crs: rasterio.crs.CRS,
    out_spatial_shape: tuple[int, int],
    out_transform: rasterio.transform.Affine,
    out_crs: rasterio.crs.CRS,
) -> Array2D[np.float32]:
    # The global source grid does not include a column for longitude 180°, so the
    # easternmost 0.25° destination pixels sit just outside the source's eastern edge
    # and bilinear resampling would leave them unset (NaN, later stored as the 0 fill
    # value). Pad the source with a wrap-around column on each side (longitude 180° ==
    # -180°) and shift the transform west by one source pixel so every destination pixel
    # falls strictly inside the source extent and interpolates across the antimeridian.
    assert abs(raw.shape[1] * src_transform.a - 360) < 1e-6, "Source must span the full globe"  # fmt: skip
    wrapped = np.concatenate([raw[:, -1:], raw, raw[:, :1]], axis=1)
    wrapped_transform = src_transform * rasterio.transform.Affine.translation(-1, 0)
    result: Array2D[np.float32]
    result, _ = rasterio.warp.reproject(
        wrapped,
        np.full(out_spatial_shape, np.nan, dtype=np.float32),
        src_transform=wrapped_transform,
        src_crs=src_crs,
        dst_transform=out_transform,
        dst_crs=out_crs,
        resampling=rasterio.warp.Resampling.bilinear,
    )
    return result
