from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyproj
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
from reformatters.noaa.hrrr.read_data import download_hrrr_file

#  All Standard Cycles go to forecast hour 18
#  Init cycles going to forecast hour 48 are 00, 06, 12, 18
EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR = pd.Series(
    {
        **{h: pd.Timedelta("48h") for h in range(0, 24, 6)},
    }
)


class NoaaHrrrForecast48HourTemplateConfig(TemplateConfig[HRRRDataVar]):
    # HRRR uses a projected coordinate system with x/y dimensions
    dims: tuple[Dim, ...] = ("init_time", "lead_time", "y", "x")
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
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "48h", freq=pd.Timedelta("1h")),
            "y": np.arange(1059),  # y-dimension for projected coordinates
            "x": np.arange(1799),  # x-dimension for projected coordinates
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """Compute non-dimension coordinates."""
        # Download a sample file to get spatial information
        filepath = download_hrrr_file(
            init_time=pd.Timestamp("2025-01-01T00:00:00"),
            lead_time=pd.Timedelta("0h"),
            domain="conus",
            file_type="sfc",
            data_vars=[
                self.data_vars[0]
            ],  # Use the first data var for bounds/resolution
        )

        hrrrds = xr.open_dataset(str(filepath), engine="rasterio")

        # Get spatial information from the HRRR file
        hrrrds_bounds = hrrrds.rio.bounds(recalc=True)
        hrrrds_res = hrrrds.rio.resolution(recalc=True)
        dx, dy = hrrrds_res[0], hrrrds_res[1]

        proj_xcorner, proj_ycorner = hrrrds_bounds[0], hrrrds_bounds[3]
        nx = ds.x.size
        ny = ds.y.size

        # Create projection coordinates
        pj = pyproj.Proj(hrrrds.rio.crs.to_proj4())
        # rio.bounds returns the lower left corner, but we want the center of the gridcell
        # so we offset by half the gridcell size.
        x_coords = (proj_xcorner + (0.5 * dx)) + np.arange(nx) * dx
        y_coords = (proj_ycorner + (0.5 * dy)) + np.arange(ny) * dy

        # Create 2D meshgrids for lat/lon conversion
        xs, ys = np.meshgrid(x_coords, y_coords)
        lons, lats = pj(xs, ys, inverse=True)

        # Calculate valid_time as init_time + lead_time
        valid_time = ds["init_time"] + ds["lead_time"]

        # Expected forecast length based on initialization hour
        expected_lengths = EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR.loc[
            ds["init_time"].dt.hour
        ]

        return {
            # Spatial reference (required by base class)
            # Valid time coordinate
            "valid_time": valid_time,
            # Forecast length metadata
            "expected_forecast_length": (("init_time",), expected_lengths.values),
            "ingested_forecast_length": (
                ("init_time",),
                expected_lengths.values,
            ),
            # Latitude and longitude as 2D coordinates over y,x
            "latitude": (("y", "x"), lats),
            "longitude": (("y", "x"), lons),
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
                    shards=None,
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
                    shards=None,
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
                    chunks=len(dim_coords["x"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="m",
                    statistics_approximate=StatisticsApproximate(
                        min=-2700000.0,
                        max=2700000.0,
                    ),
                ),
            ),
            Coordinate(
                name="y",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["y"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="m",
                    statistics_approximate=StatisticsApproximate(
                        min=-1600000.0,
                        max=1600000.0,
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
                    shards=None,
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
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            # Add latitude and longitude as non-dimension coordinates
            Coordinate(
                name="latitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=(len(dim_coords["y"]), len(dim_coords["x"])),
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
                    chunks=(len(dim_coords["y"]), len(dim_coords["x"])),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="degrees_east",
                    statistics_approximate=StatisticsApproximate(
                        min=-134.09548,
                        max=-60.917192,
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
            "x": 180,
            "y": 180,
        }

        # TODO: About XXXMB compressed, about XXGB uncompressed
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 49,
            "x": 180 * 3,  # 3 shards
            "y": 180 * 3,  # 3 shards
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
