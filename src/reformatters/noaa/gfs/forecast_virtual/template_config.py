from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
    StatisticsApproximate,
    prepend_comment,
)
from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, Dims, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.gfs.virtual_template_config import NoaaGfsVirtualTemplateConfig
from reformatters.noaa.models import NoaaDataVar

FORECAST_LENGTH = pd.Timedelta("384h")

_BUCKET_EXTENT = (
    "the preceding 1-6 hours of forecast lead time, since the last lead time divisible "
    "by 6 before this step."
)
_BUCKET_COMMENTS = {
    "accum": (
        f"Accumulated over {_BUCKET_EXTENT} Subtracting the value at an earlier lead "
        "time with the same window start gives the exact total between those two lead "
        "times."
    ),
    "avg": f"Averaged over {_BUCKET_EXTENT}",
    "max": f"Maximum value over {_BUCKET_EXTENT}",
    "min": f"Minimum value over {_BUCKET_EXTENT}",
}
_RUN_TOTAL_COMMENT = (
    "Accumulated from the forecast initialization time to this step, so the window "
    "lengthens with lead time and never resets. Subtracting the value at an earlier "
    "step gives the exact total between those two steps."
)


class NoaaGfsForecastVirtualTemplateConfig(NoaaGfsVirtualTemplateConfig):
    """Virtual GFS forecast on init_time x lead_time, one chunk per GRIB message."""

    dims: Dims = {
        ROOT: ("init_time", "lead_time", "latitude", "longitude"),
        "pressure_level": (
            "init_time",
            "lead_time",
            "latitude",
            "longitude",
            "pressure_level",
        ),
        "height_above_mean_sea_level": (
            "init_time",
            "lead_time",
            "latitude",
            "longitude",
            "height_above_mean_sea_level",
        ),
    }
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2021-05-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-gfs-forecast-virtual",
            dataset_version="0.1.0",
            name="NOAA GFS forecast, virtual",
            description="Weather forecasts from the Global Forecast System (GFS) operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP GFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {whole_hours(self.append_dim_frequency)} hours",
            forecast_domain="Forecast lead time 0-384 hours (0-16 days) ahead",
            forecast_resolution="Forecast step 0-120 hours: hourly, 123-384 hours: 3 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "120h", freq="1h").union(
                pd.timedelta_range("123h", FORECAST_LENGTH, freq="3h")
            ),
            **self._latitude_longitude_coordinates(),
            **self._vertical_dimension_coordinates(),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "expected_forecast_length": (
                ("init_time",),
                np.full(
                    ds["init_time"].size,
                    FORECAST_LENGTH.to_timedelta64(),
                    dtype="timedelta64[us]",
                ),
            ),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return [
            *super().coords,
            Coordinate(
                name="init_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Forecast initialization time",
                    standard_name="forecast_reference_time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present"
                    ),
                ),
            ),
            Coordinate(
                name="lead_time",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=len(dim_coords["lead_time"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Forecast lead time",
                    standard_name="forecast_period",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            *self._vertical_coords(),
            Coordinate(
                name="valid_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=(
                        append_dim_coordinate_chunk_size,
                        len(dim_coords["lead_time"]),
                    ),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Valid time",
                    standard_name="time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(),
                        max=f"Present + {whole_hours(FORECAST_LENGTH)} hours",
                    ),
                ),
            ),
            Coordinate(
                name="expected_forecast_length",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Expected forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(FORECAST_LENGTH), max=str(FORECAST_LENGTH)
                    ),
                ),
            ),
        ]

    def _catalog_data_vars(self) -> list[NoaaDataVar]:
        """The shared catalog with each windowed variable's window described in
        forecast lead time."""
        return [_with_window_comment(var) for var in super()._catalog_data_vars()]


def _with_window_comment(var: NoaaDataVar) -> NoaaDataVar:
    reset_frequency = var.internal_attrs.window_reset_frequency
    if reset_frequency is None:
        return var
    return prepend_comment(
        var,
        _RUN_TOTAL_COMMENT
        if reset_frequency == pd.Timedelta.max
        else _BUCKET_COMMENTS[var.attrs.step_type],
    )
