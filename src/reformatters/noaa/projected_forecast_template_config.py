from collections.abc import Sequence
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DataVar,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, Dims, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.conus_grid import NoaaConusTemplateConfig

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar)


class NoaaProjectedForecastTemplateConfig(
    NoaaConusTemplateConfig[DATA_VAR], Generic[DATA_VAR]
):
    forecast_length: Timedelta
    first_lead: Timedelta = pd.Timedelta("0h")
    lead_frequency: Timedelta = pd.Timedelta("1h")
    dims: Dims = {ROOT: ("init_time", "lead_time", "y", "x")}
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp

    def _vertical_dimension_coordinates(self) -> dict[str, Any]:
        return {}

    def _vertical_coords(self) -> list[Coordinate]:
        return []

    def _expected_forecast_length_values(self, ds: xr.Dataset) -> np.ndarray[Any, Any]:
        """Per-init expected forecast length; a constant forecast_length unless overridden."""
        return np.full(ds[self.append_dim].size, self.forecast_length.to_timedelta64())

    def _expected_forecast_length_statistics(self) -> StatisticsApproximate:
        return StatisticsApproximate(
            min=str(self.forecast_length), max=str(self.forecast_length)
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        y_coords, x_coords = self._y_x_coordinates()
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range(
                self.first_lead, self.forecast_length, freq=self.lead_frequency
            ),
            "y": y_coords,
            "x": x_coords,
            **self._vertical_dimension_coordinates(),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        latitudes, longitudes = self._latitude_longitude_coordinates(
            ds["x"].values, ds["y"].values
        )
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "expected_forecast_length": (
                ("init_time",),
                self._expected_forecast_length_values(ds),
            ),
            "latitude": (("y", "x"), latitudes),
            "longitude": (("y", "x"), longitudes),
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
                    fill_value=float("nan"),
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
                        max=f"Present + {whole_hours(self.forecast_length)} hours",
                    ),
                ),
            ),
            Coordinate(
                name="expected_forecast_length",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=float("nan"),
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Expected forecast length",
                    units="seconds",
                    statistics_approximate=self._expected_forecast_length_statistics(),
                ),
            ),
        ]
