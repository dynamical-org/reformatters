from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.noaa.hrrr.hrrr_config_models import HRRRDataVar, HRRRInternalAttrs

#  All Standard Cycles go to forecast hour 18
#  Init cycles going to forecast hour 48 are 00, 06, 12, 18
EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR = pd.Series(
    {
        **{h: pd.Timedelta("18h") for h in range(24)},
        **{
            h: pd.Timedelta("48h") for h in range(0, 24, 6)
        },  # must be splatted last to overwrite 18h values
    }
)


class NoaaHrrrForecast48HourTemplateConfig(TemplateConfig[HRRRDataVar]):
    # HRRR uses a projected coordinate system, but we'll use latitude/longitude
    # as dimension names to conform with the common interface while
    # maintaining x/y as non-dimension coordinates
    dims: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2018-07-13T12:00")  # start of HRRR v3
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-hrrr-forecast-48-hour",
            dataset_version="0.0.0",
            name="NOAA HRRR forecast, 48 hour",
            description="Weather forecasts from the High Resolution Rapid Refresh (HRRR) model operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP HRRR data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            spatial_domain="CONUS",
            spatial_resolution="3km",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution="Forecasts initialized every 6 hours.",
            forecast_domain="Forecast lead time 0-48 hours ahead",
            forecast_resolution="Hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        # HRRR uses projected coordinates but we map them to lat/lon dimensions
        # TODO: These should be actual latitude/longitude arrays for the HRRR grid
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "48h", freq=pd.Timedelta("1h")),
            "latitude": np.arange(1059),  # y-dimension mapped to latitude
            "longitude": np.arange(1799),  # x-dimension mapped to longitude
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """Compute non-dimension coordinates."""
        # Calculate valid_time as init_time + lead_time
        valid_time = ds["init_time"] + ds["lead_time"]

        # Calculate expected and ingested forecast lengths
        init_hours = ds["init_time"].dt.hour
        expected_lengths = xr.zeros_like(init_hours, dtype="timedelta64[ns]")
        for hour in range(24):
            if hour in [0, 6, 12, 18]:
                expected_lengths = expected_lengths.where(
                    init_hours != hour, pd.Timedelta("48h")
                )
            else:
                expected_lengths = expected_lengths.where(
                    init_hours != hour, pd.Timedelta("18h")
                )

        # Add x/y projected coordinates
        x_coords = np.arange(1799)  # Original x coordinates
        y_coords = np.arange(1059)  # Original y coordinates

        return {
            **super().derive_coordinates(ds),
            "valid_time": valid_time,
            "expected_forecast_length": (("init_time",), expected_lengths.data),
            "ingested_forecast_length": (
                ("init_time",),
                expected_lengths.data,
            ),  # Initial placeholder
            "x": (("longitude",), x_coords),  # Map longitude dim to x coordinates
            "y": (("latitude",), y_coords),  # Map latitude dim to y coordinates
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return [
            Coordinate(
                name="init_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=append_dim_coordinate_chunk_size,
                ),
                attrs=CoordinateAttrs(
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present"
                    ),
                ),
            ),
            Coordinate(
                name="lead_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=len(dim_coords["lead_time"]),
                    shards=len(dim_coords["lead_time"]),
                ),
                attrs=CoordinateAttrs(
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="x",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["longitude"]),
                    shards=len(dim_coords["longitude"]),
                ),
                attrs=CoordinateAttrs(
                    units="unitless",
                    statistics_approximate=StatisticsApproximate(
                        min=0,
                        max=1798,
                    ),
                ),
            ),
            Coordinate(
                name="y",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["latitude"]),
                    shards=len(dim_coords["latitude"]),
                ),
                attrs=CoordinateAttrs(
                    units="unitless",
                    statistics_approximate=StatisticsApproximate(
                        min=0,
                        max=1058,
                    ),
                ),
            ),
            Coordinate(
                name="latitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["latitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="degrees_north",
                    statistics_approximate=StatisticsApproximate(
                        min=21.138123,
                        max=52.615653,
                    ),
                ),
            ),
            Coordinate(
                name="longitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["longitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="degrees_east",
                    # TODO: How to set these min/max values?
                    statistics_approximate=StatisticsApproximate(
                        min=-134.09548,
                        max=-60.917192,
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
                    shards=(
                        append_dim_coordinate_chunk_size,
                        len(dim_coords["lead_time"]),
                    ),
                ),
                attrs=CoordinateAttrs(
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present + 48 hours"
                    ),
                ),
            ),
            Coordinate(
                name="ingested_forecast_length",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=append_dim_coordinate_chunk_size,
                ),
                attrs=CoordinateAttrs(
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
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=append_dim_coordinate_chunk_size,
                ),
                attrs=CoordinateAttrs(
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    chunks=(),  # Scalar coordinate
                    shards=(),
                ),
                attrs=CoordinateAttrs(
                    units="unitless",
                    statistics_approximate=StatisticsApproximate(
                        min=0,
                        max=0,
                    ),
                ),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[HRRRDataVar]:
        # TODO: These chunks are about XXXmb of uncompressed float32s
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 49,
            "latitude": 180,
            "longitude": 180,
        }

        # TODO: About XXXMB compressed, about XXGB uncompressed
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 49,
            "latitude": 180 * 3,  # 2 shards
            "longitude": 180 * 3,  # 4 shards
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        return [
            HRRRDataVar(
                name="composite_reflectivity",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="refc",
                    long_name="Composite reflectivity",
                    units="dBZ",
                    step_type="instant",
                ),
                internal_attrs=HRRRInternalAttrs(
                    grib_element="REFC",
                    grib_description="REFC:entire atmosphere",  # TODO
                    index_position=0,
                    keep_mantissa_bits=10,
                    grib_index_level="entire atmosphere",
                    hrrr_file_type="sfc",
                ),
            )
        ]
