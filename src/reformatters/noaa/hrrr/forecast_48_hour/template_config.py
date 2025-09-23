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
from reformatters.common.template_config import SPATIAL_REF_COORDS, TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
    NoaaHrrrInternalAttrs,
)

#  All Standard Cycles go to forecast hour 18
#  Init cycles going to forecast hour 48 are 00, 06, 12, 18
EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR = pd.Series(
    {
        **{h: pd.Timedelta("48h") for h in range(0, 24, 6)},
    }
)


class NoaaHrrrForecast48HourTemplateConfig(TemplateConfig[NoaaHrrrDataVar]):
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
            dataset_version="0.1.0",
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
        # Calculate x and y coordinates
        shape, bounds, resolution, _crs = self._spatial_info()
        dx, dy = resolution
        left, _bottom, _right, top = bounds
        ny, nx = shape
        # add 1/2 a pixel to corner of bounds to get pixel center
        y_coords = (top + (0.5 * dy)) + (np.arange(ny) * dy)
        x_coords = (left + (0.5 * dx)) + (np.arange(nx) * dx)

        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "48h", freq=pd.Timedelta("1h")),
            "y": y_coords,
            "x": x_coords,
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """Compute non-dimension coordinates."""
        # Create 2D latitude and longitude grids
        # x, y are the spatial dimensions of this dataset
        # latitude and longitude there
        _shape, _bounds, _resolution, crs = self._spatial_info()
        xs, ys = np.meshgrid(ds["x"], ds["y"])
        lons, lats = pyproj.Proj(crs)(xs, ys, inverse=True)
        # Dropping to 32 bit precision still gets us < 1 meter precision and
        # makes each array about 6MB vs 15MB for float64.
        lats = lats.astype(np.float32)
        lons = lons.astype(np.float32)

        # Expected forecast length based on initialization hour
        expected_lengths = EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR.loc[
            ds[self.append_dim].dt.hour.values
        ]

        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "expected_forecast_length": (
                ("init_time",),
                np.asarray(expected_lengths.values),
            ),
            "ingested_forecast_length": (
                ("init_time",),
                np.full(ds[self.append_dim].size, np.timedelta64("NaT", "ns")),
            ),
            "latitude": (("y", "x"), lats),
            "longitude": (("y", "x"), lons),
            "spatial_ref": SPATIAL_REF_COORDS,
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
                    dtype="float32",
                    fill_value=np.nan,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
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
                    dtype="float32",
                    fill_value=np.nan,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
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
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    chunks=(),  # Scalar coordinate
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    # Derived from opening a sample HRRR file, see
                    # tests/noaa/hrrr/forecast_48_hour/template_config_test.py::test_spatial_info_matches_file
                    GeoTransform="-2699020.142521929 3000.0 0.0 1588193.847443335 0.0 -3000.0",
                    crs_wkt='PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",38.5],PARAMETER["central_meridian",-97.5],PARAMETER["standard_parallel_1",38.5],PARAMETER["standard_parallel_2",38.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]',
                    false_easting=0.0,
                    false_northing=0.0,
                    geographic_crs_name="Coordinate System imported from GRIB file",
                    grid_mapping_name="lambert_conformal_conic",
                    horizontal_datum_name="unnamed",
                    inverse_flattening=0.0,
                    latitude_of_projection_origin=38.5,
                    longitude_of_central_meridian=-97.5,
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    projected_crs_name="unnamed",
                    reference_ellipsoid_name="Sphere",
                    semi_major_axis=6371229.0,
                    semi_minor_axis=6371229.0,
                    spatial_ref='PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",38.5],PARAMETER["central_meridian",-97.5],PARAMETER["standard_parallel_1",38.5],PARAMETER["standard_parallel_2",38.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]',
                    standard_parallel=(38.5, 38.5),
                ),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[NoaaHrrrDataVar]:
        # ~15.6MB uncompressed, ~3.1MB compressed
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 49,  # all lead times
            "x": 300,  # 6 chunks (1799 pixels)
            "y": 265,  # 4 chunks (1059 pixels)
        }

        # Single shard for each init time
        # ~374MB uncompressed, ~75MB compressed
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 49,
            "x": var_chunks["x"] * 6,  # single shard
            "y": var_chunks["y"] * 4,  # single shard
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_window_reset_frequency = pd.Timedelta("1h")
        default_keep_mantissa_bits = 7

        return [
            NoaaHrrrDataVar(
                name="composite_reflectivity",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="refc",
                    long_name="Composite reflectivity",
                    units="dBZ",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="REFC",
                    grib_description='0[-] EATM="Entire Atmosphere"',
                    index_position=1,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    grib_index_level="entire atmosphere",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="t2m",
                    long_name="2 metre temperature",
                    units="C",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="TMP",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=71,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="wind_u_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="u10",
                    long_name="10 metre U wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="UGRD",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_index_level="10 m above ground",
                    index_position=77,
                    keep_mantissa_bits=6,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="wind_v_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="v10",
                    long_name="10 metre V wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="VGRD",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_index_level="10 m above ground",
                    index_position=78,
                    keep_mantissa_bits=6,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tp",
                    long_name="Total Precipitation",
                    units="mm/s",
                    comment="Average precipitation rate since the previous forecast step.",
                    step_type="avg",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="APCP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=84,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="precipitable_water_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="pwat",
                    long_name="Precipitable water",
                    units="kg/(m^2)",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="PWAT",
                    grib_description='0[-] EATM="Entire atmosphere (considered as a single layer)"',
                    grib_index_level="entire atmosphere (considered as a single layer)",
                    index_position=107,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="total_cloud_cover_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tcc",
                    long_name="Total Cloud Cover",
                    units="%",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="TCDC",
                    grib_description='0[-] EATM="Entire Atmosphere"',
                    grib_index_level="entire atmosphere",
                    index_position=116,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="downward_short_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    long_name="Surface downward short-wave radiation flux",
                    units="W/(m^2)",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="DSWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=123,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="downward_long_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    long_name="Surface downward long-wave radiation flux",
                    units="W/(m^2)",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="DLWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=124,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    long_name="Pressure reduced to MSL",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="MSLMA",
                    grib_description='0[-] MSL="Mean sea level"',
                    grib_index_level="mean sea level",
                    index_position=41,
                    keep_mantissa_bits=13,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="percent_frozen_precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cpofp",
                    long_name="Percent frozen precipitation",
                    units="%",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CPOFP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=82,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="pressure_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sp",
                    long_name="Surface pressure",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="PRES",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=62,
                    keep_mantissa_bits=10,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="categorical_ice_pellets_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cicep",
                    long_name="Categorical ice pellets",
                    units="0=no; 1=yes",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CICEP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=91,
                    keep_mantissa_bits="no-rounding",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="categorical_snow_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="csnow",
                    long_name="Categorical snow",
                    units="0=no; 1=yes",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CSNOW",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=90,
                    keep_mantissa_bits="no-rounding",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="categorical_freezing_rain_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cfrzr",
                    long_name="Categorical freezing rain",
                    units="0=no; 1=yes",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CFRZR",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=92,
                    keep_mantissa_bits="no-rounding",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="categorical_rain_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="crain",
                    long_name="Categorical rain",
                    units="0=no; 1=yes",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CRAIN",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=93,
                    keep_mantissa_bits="no-rounding",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="relative_humidity_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="r2",
                    long_name="2 metre relative humidity",
                    units="%",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="RH",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=75,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="geopotential_height_cloud_ceiling",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="gh",
                    long_name="Geopotential height",
                    units="gpm",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="HGT",
                    grib_description='0[-] CEIL="Cloud ceiling"',
                    grib_index_level="cloud ceiling",
                    index_position=117,
                    keep_mantissa_bits=8,
                    hrrr_file_type="sfc",
                ),
            ),
            # HRRR provides 80m but not 100m winds
            NoaaHrrrDataVar(
                name="wind_u_80m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="u80",
                    long_name="U-component of wind (80 m above ground)",
                    units="m/s",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="UGRD",
                    grib_description='80[m] HTGL="Specified height level above ground"',
                    grib_index_level="80 m above ground",
                    index_position=60,
                    keep_mantissa_bits=6,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="wind_v_80m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="v80",
                    long_name="V-component of wind (80 m above ground)",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="VGRD",
                    grib_description='80[m] HTGL="Specified height level above ground"',
                    grib_index_level="80 m above ground",
                    index_position=61,
                    keep_mantissa_bits=6,
                    hrrr_file_type="sfc",
                ),
            ),
        ]

    def _spatial_info(
        self,
    ) -> tuple[
        tuple[int, int], tuple[float, float, float, float], tuple[float, float], str
    ]:
        """
        Returns (shape, bounds, resolution, crs proj4 string).
        Useful for deriving x, y and latitude, longitude coordinates.
        See tests/noaa/hrrr/forecast_48_hour/template_config_test.py::test_spatial_info_matches_file
        """
        return (
            (1059, 1799),
            (
                -2699020.142521929,
                -1588806.152556665,
                2697979.857478071,
                1588193.847443335,
            ),
            (3000.0, -3000.0),
            "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs=True",
        )
