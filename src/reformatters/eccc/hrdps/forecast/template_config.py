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
    Group,
    StatisticsApproximate,
)
from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.eccc.hrdps.hrdps_config_models import EcccHrdpsDataVar
from reformatters.eccc.hrdps.template_config import (
    HRDPS_INIT_FREQUENCY,
    EcccHrdpsCommonTemplateConfig,
)


class EcccHrdpsForecastTemplateConfig(EcccHrdpsCommonTemplateConfig):
    dims: dict[Group, tuple[Dim, ...]] = {ROOT: ("init_time", "lead_time", "y", "x")}
    append_dim: AppendDim = "init_time"
    # Start of dynamical.org's continuous grib archive of HRDPS on Source Co-Op
    append_dim_start: Timestamp = pd.Timestamp("2026-07-09T00:00")
    append_dim_frequency: Timedelta = HRDPS_INIT_FREQUENCY

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="eccc-hrdps-forecast",
            dataset_version="0.1.0",
            name="ECCC HRDPS forecast",
            description="Weather forecasts from the High Resolution Deterministic "
            "Prediction System (HRDPS) continental domain, operated by Environment "
            "and Climate Change Canada (ECCC).",
            attribution="Data Source: Environment and Climate Change Canada. "
            "Processed by dynamical.org.",
            license="ECCC Data Servers End-use Licence v2.1",
            spatial_domain="Canada and the northern continental United States",
            spatial_resolution="2.5 km",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution="Forecasts initialized every 6 hours",
            forecast_domain="Forecast lead time 0-48 hours ahead",
            forecast_resolution="Hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        y_coords, x_coords = self._y_x_coordinates()
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "48h", freq="1h"),
            "y": y_coords,
            "x": x_coords,
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
                np.full(
                    ds[self.append_dim].size,
                    ds["lead_time"].max(),
                    dtype="timedelta64[us]",
                ),
            ),
            "ingested_forecast_length": (
                ("init_time",),
                np.full(ds[self.append_dim].size, np.timedelta64("NaT", "us")),
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
                        max="Present + 48 hours",
                    ),
                ),
            ),
            Coordinate(
                name="ingested_forecast_length",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=float("nan"),
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Ingested forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
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
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[EcccHrdpsDataVar]:
        # ~12MB uncompressed, ~2.5MB compressed
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 49,  # all lead times
            "y": 258,  # 5 chunks over 1290 pixels
            "x": 254,  # 10 chunks over 2540 pixels
        }

        # Single shard per init time
        # ~612MB uncompressed, ~122MB compressed
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 49,
            "y": var_chunks["y"] * 5,  # single shard
            "x": var_chunks["x"] * 10,  # single shard
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims[ROOT]),
            shards=tuple(var_shards[d] for d in self.dims[ROOT]),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        return self.get_data_vars(encoding_float32_default)
