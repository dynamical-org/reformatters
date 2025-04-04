from collections.abc import Sequence
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
    replace,
)
from reformatters.common.types import DatetimeLike
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.noaa.noaa_config_models import NOAADataVar, NOAAInternalAttrs

DATASET_ID = "noaa-gfs-forecast"
DATASET_VERSION = "0.0.1-dev"

# Technically there is data going back to 2021-04-13 but we are choosing a round date to start with
INIT_TIME_START = pd.Timestamp("2021-05-01T00:00")
INIT_TIME_FREQUENCY = pd.Timedelta("6h")

DATASET_ATTRIBUTES = DatasetAttributes(
    dataset_id=DATASET_ID,
    dataset_version=DATASET_VERSION,
    name="NOAA GFS forecast",
    description="Weather forecasts from the Global Forecast System (GFS) operated by NOAA NWS NCEP.",
    attribution="NOAA NWS NCEP GFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
    spatial_domain="Global",
    spatial_resolution="0.25 degrees (~20km)",
    time_domain=f"Forecasts initialized {INIT_TIME_START} UTC to Present",
    time_resolution=f"Forecasts initialized every {int(INIT_TIME_FREQUENCY.total_seconds() / 3600)} hours.",
    forecast_domain="Forecast lead time 0-384 hours (0-16 days) ahead",
    forecast_resolution="Forecast step 0-120 hours: hourly, 123-384 hours: 3 hourly",
)

# Silly to list dims twice, but typing.get_args() doesn't guarantee the return order,
# the order in DIMS is important, and type parameters can't be constants.
type Dim =           Literal["init_time", "lead_time", "latitude", "longitude"]  # fmt: off
VAR_DIMS: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")  # fmt: off

APPEND_DIMENSION = "init_time"


def get_template_dimension_coordinates() -> dict[str, Any]:
    return {
        "init_time": get_init_time_coordinates(INIT_TIME_START + INIT_TIME_FREQUENCY),
        "lead_time": pd.timedelta_range("0h", "120h", freq="1h").union(
            pd.timedelta_range("123h", "384h", freq="3h")
        ),
        # latitude descends when north is up
        "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
        "longitude": np.arange(-180, 180, 0.25),
    }


def get_init_time_coordinates(
    init_time_end: DatetimeLike,
) -> pd.DatetimeIndex:
    return pd.date_range(
        INIT_TIME_START, init_time_end, freq=INIT_TIME_FREQUENCY, inclusive="left"
    )


# CHUNKS
VAR_CHUNKS: dict[Dim, int] = {
    "init_time": 1,  # one forecast per chunk
    "lead_time": 105,  # 2 chunks
    "latitude": 121,  # 6 chunks over 721 pixels
    "longitude": 121,  # 12 chunks over 1440 pixels
}

# SHARDS
VAR_SHARDS: dict[Dim, int] = {
    "init_time": 1,  # one forecast per shard
    "lead_time": VAR_CHUNKS["lead_time"] * 2,  # all lead times in one shard
    "latitude": VAR_CHUNKS["latitude"] * 6,  # all latitudes in one shard
    "longitude": VAR_CHUNKS["longitude"] * 6,  # all longitudes in one shard
}

VAR_CHUNKS_ORDERED = tuple(VAR_CHUNKS[dim] for dim in VAR_DIMS)
VAR_SHARDS_ORDERED = tuple(VAR_SHARDS[dim] for dim in VAR_DIMS)

# The init time dimension is our append dimension during updates.
# We also want coordinates to be in a single chunk for dataset open speed.
# By fixing the chunk size for coordinates along the append dimension to
# something much larger than we will really use, the array is always
# a fixed underlying chunk size and values in it can be safely updated
# prior to metadata document updates that increase the reported array size.
# This is a zarr format hack to allow expanding an array safely and requires
# that new array values are written strictly before new metadata is written
# (doing this correctly is a key benefit of icechunk).
INIT_TIME_COORDINATE_CHUNK_SIZE = int(pd.Timedelta(days=365 * 15) / INIT_TIME_FREQUENCY)

GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT: Final[int] = 7
GFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL: Final[Literal["no-rounding"]] = (
    "no-rounding"
)

ENCODING_FLOAT32_DEFAULT = Encoding(
    dtype="float32",
    fill_value=np.nan,
    chunks=VAR_CHUNKS_ORDERED,
    shards=VAR_SHARDS_ORDERED,
    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
)

_dim_coords = get_template_dimension_coordinates()

COORDINATES: Sequence[Coordinate] = (
    Coordinate(
        name="init_time",
        encoding=Encoding(
            dtype="int64",
            fill_value=0,
            compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
            calendar="proleptic_gregorian",
            units="seconds since 1970-01-01 00:00:00",
            chunks=INIT_TIME_COORDINATE_CHUNK_SIZE,
            shards=INIT_TIME_COORDINATE_CHUNK_SIZE,
        ),
        attrs=CoordinateAttrs(
            units="seconds since 1970-01-01 00:00:00",
            statistics_approximate=StatisticsApproximate(
                min=INIT_TIME_START.isoformat(), max="Present"
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
            chunks=len(_dim_coords["lead_time"]),
            shards=len(_dim_coords["lead_time"]),
        ),
        attrs=CoordinateAttrs(
            units="seconds",
            statistics_approximate=StatisticsApproximate(
                min=str(_dim_coords["lead_time"].min()),
                max=str(_dim_coords["lead_time"].max()),
            ),
        ),
    ),
    Coordinate(
        name="latitude",
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
            chunks=len(_dim_coords["latitude"]),
            shards=len(_dim_coords["latitude"]),
        ),
        attrs=CoordinateAttrs(
            units="degrees_north",
            statistics_approximate=StatisticsApproximate(
                min=_dim_coords["latitude"].min(),
                max=_dim_coords["latitude"].max(),
            ),
        ),
    ),
    Coordinate(
        name="longitude",
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
            chunks=len(_dim_coords["longitude"]),
            shards=len(_dim_coords["longitude"]),
        ),
        attrs=CoordinateAttrs(
            units="degrees_east",
            statistics_approximate=StatisticsApproximate(
                min=_dim_coords["longitude"].min(),
                max=_dim_coords["longitude"].max(),
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
            chunks=(INIT_TIME_COORDINATE_CHUNK_SIZE, len(_dim_coords["lead_time"])),
            shards=(INIT_TIME_COORDINATE_CHUNK_SIZE, len(_dim_coords["lead_time"])),
        ),
        attrs=CoordinateAttrs(
            units="seconds since 1970-01-01 00:00:00",
            statistics_approximate=StatisticsApproximate(
                min=INIT_TIME_START.isoformat(), max="Present + 16 days"
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
            chunks=INIT_TIME_COORDINATE_CHUNK_SIZE,
            shards=INIT_TIME_COORDINATE_CHUNK_SIZE,
        ),
        attrs=CoordinateAttrs(
            units="seconds",
            statistics_approximate=StatisticsApproximate(
                min=str(_dim_coords["lead_time"].min()),
                max=str(_dim_coords["lead_time"].max()),
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
            chunks=INIT_TIME_COORDINATE_CHUNK_SIZE,
            shards=INIT_TIME_COORDINATE_CHUNK_SIZE,
        ),
        attrs=CoordinateAttrs(
            units="seconds",
            statistics_approximate=StatisticsApproximate(
                min=str(_dim_coords["lead_time"].min()),
                max=str(_dim_coords["lead_time"].max()),
            ),
        ),
    ),
    Coordinate(
        name="spatial_ref",
        encoding=Encoding(
            dtype="int64",
            fill_value=0,
            chunks=1,  # Scalar coordinate
            shards=1,
        ),
        attrs=CoordinateAttrs(
            units="unitless",
            statistics_approximate=StatisticsApproximate(
                min=0,
                max=0,
            ),
        ),
    ),
)

# index_position = BAND - 1 from the grib files
DATA_VARIABLES: Sequence[NOAADataVar] = (
    NOAADataVar(
        name="pressure_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="sp",
            long_name="Surface pressure",
            units="Pa",
            step_type="instant",
            standard_name="surface_air_pressure",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="PRES",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            index_position=560,
            keep_mantissa_bits=10,
        ),
    ),
    NOAADataVar(
        name="temperature_2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="t2m",
            long_name="2 metre temperature",
            units="C",
            step_type="instant",
            standard_name="air_temperature",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="TMP",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            index_position=580,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="relative_humidity_2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="r2",
            long_name="2 metre relative humidity",
            units="%",
            step_type="instant",
            standard_name="relative_humidity",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="RH",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            index_position=583,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="maximum_temperature_2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="tmax",
            long_name="Maximum temperature",
            units="C",
            step_type="max",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="TMAX",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            index_position=585,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="minimum_temperature_2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="tmin",
            long_name="Minimum temperature",
            units="C",
            step_type="min",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="TMIN",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            index_position=586,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="wind_u_10m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="u10",
            long_name="10 metre U wind component",
            units="m/s",
            step_type="instant",
            standard_name="eastward_wind",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="UGRD",
            grib_description='10[m] HTGL="Specified height level above ground"',
            grib_index_level="10 m above ground",
            index_position=587,
            keep_mantissa_bits=6,
        ),
    ),
    NOAADataVar(
        name="wind_v_10m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="v10",
            long_name="10 metre V wind component",
            units="m/s",
            step_type="instant",
            standard_name="northward_wind",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="VGRD",
            grib_description='10[m] HTGL="Specified height level above ground"',
            grib_index_level="10 m above ground",
            index_position=588,
            keep_mantissa_bits=6,
        ),
    ),
    NOAADataVar(
        name="wind_u_100m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="u100",
            long_name="100 metre U wind component",
            standard_name="eastward_wind",
            units="m/s",
            step_type="instant",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="UGRD",
            grib_description='100[m] HTGL="Specified height level above ground"',
            grib_index_level="100 m above ground",
            index_position=688,
            keep_mantissa_bits=6,
        ),
    ),
    NOAADataVar(
        name="wind_v_100m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="v100",
            long_name="100 metre V wind component",
            units="m/s",
            step_type="instant",
            standard_name="northward_wind",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="VGRD",
            grib_index_level="100 m above ground",
            grib_description='100[m] HTGL="Specified height level above ground"',
            index_position=689,
            keep_mantissa_bits=6,
        ),
    ),
    NOAADataVar(
        name="percent_frozen_precipitation_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="cpofp",
            long_name="Percent frozen precipitation",
            units="%",
            step_type="instant",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="CPOFP",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            index_position=590,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="precipitation_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="tp",
            long_name="Total Precipitation",
            units="mm/s",
            comment="Average precipitation rate since the previous forecast step.",
            step_type="avg",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="APCP",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            index_position=595,
            include_lead_time_suffix=True,
            deaccumulate_to_rates=True,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="categorical_snow_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="csnow",
            long_name="Categorical snow",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="CSNOW",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            index_position=604,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL,
        ),
    ),
    NOAADataVar(
        name="categorical_ice_pellets_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="cicep",
            long_name="Categorical ice pellets",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="CICEP",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            index_position=605,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL,
        ),
    ),
    NOAADataVar(
        name="categorical_freezing_rain_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="cfrzr",
            long_name="Categorical freezing rain",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="CFRZR",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            index_position=606,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL,
        ),
    ),
    NOAADataVar(
        name="categorical_rain_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="crain",
            long_name="Categorical rain",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="CRAIN",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            index_position=607,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL,
        ),
    ),
    NOAADataVar(
        name="precipitable_water_atmosphere",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="pwat",
            long_name="Precipitable water",
            units="kg/(m^2)",
            step_type="instant",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="PWAT",
            grib_description='0[-] EATM="Entire atmosphere (considered as a single layer)"',
            grib_index_level="entire atmosphere (considered as a single layer)",
            index_position=625,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="total_cloud_cover_atmosphere",
        encoding=replace(ENCODING_FLOAT32_DEFAULT),
        attrs=DataVarAttrs(
            short_name="tcc",
            long_name="Total Cloud Cover",
            units="%",
            step_type="avg",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="TCDC",
            grib_description='0[-] EATM="Entire Atmosphere"',
            grib_index_level="entire atmosphere",
            index_position=635,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="geopotential_height_cloud_ceiling",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="gh",
            long_name="Geopotential height",
            units="gpm",
            step_type="instant",
            standard_name="geopotential_height",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="HGT",
            grib_description='0[-] CEIL="Cloud ceiling"',
            grib_index_level="cloud ceiling",
            index_position=637,
            keep_mantissa_bits=8,
        ),
    ),
    NOAADataVar(
        name="downward_short_wave_radiation_flux_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="sdswrf",
            long_name="Surface downward short-wave radiation flux",
            units="W/(m^2)",
            step_type="avg",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="DSWRF",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            index_position=652,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="downward_long_wave_radiation_flux_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="sdlwrf",
            long_name="Surface downward long-wave radiation flux",
            units="W/(m^2)",
            step_type="avg",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="DLWRF",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            index_position=653,
            keep_mantissa_bits=GFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    NOAADataVar(
        name="pressure_reduced_to_mean_sea_level",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            step_type="instant",
        ),
        internal_attrs=NOAAInternalAttrs(
            grib_element="PRMSL",
            grib_description='0[-] MSL="Mean sea level"',
            grib_index_level="mean sea level",
            index_position=0,
            keep_mantissa_bits=10,
        ),
    ),
)
