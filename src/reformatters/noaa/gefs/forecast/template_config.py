from collections.abc import Sequence
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
import zarr

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
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar, GEFSInternalAttrs

DATASET_ID = "noaa-gefs-forecast"
DATASET_VERSION = "0.1.0"

INIT_TIME_START = pd.Timestamp("2024-01-01T00:00")
INIT_TIME_FREQUENCY = pd.Timedelta("24h")

DATASET_ATTRIBUTES = DatasetAttributes(
    dataset_id=DATASET_ID,
    dataset_version=DATASET_VERSION,
    name="NOAA GEFS forecast",
    description="Weather forecasts from the Global Ensemble Forecast System (GEFS) operated by NOAA NWS NCEP.",
    attribution="NOAA NWS NCEP GEFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
    spatial_domain="Global",
    spatial_resolution="0-240 hours: 0.25 degrees (~20km), 243-840 hours: 0.5 degrees (~40km)",
    time_domain=f"Forecasts initialized {INIT_TIME_START} UTC to Present",
    time_resolution="Forecasts initialized every 24 hours.",
    forecast_domain="Forecast lead time 0-840 hours (0-35 days) ahead",
    forecast_resolution="Forecast step 0-240 hours: 3 hourly, 243-840 hours: 6 hourly",
)

# Silly to list dims twice, but typing.get_args() doesn't guarantee the return order,
# the order in DIMS is important, and type parameters can't be constants.
type Dim =                     Literal["init_time", "ensemble_member", "lead_time", "latitude", "longitude"]  # fmt: off
ENSEMBLE_VAR_DIMS: tuple[Dim, ... ] = ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")  # fmt: off
STATISTIC_VAR_DIMS: tuple[Dim, ...] = ("init_time",                    "lead_time", "latitude", "longitude")  # fmt: off


def get_template_dimension_coordinates() -> dict[Dim, Any]:
    return {
        "init_time": get_init_time_coordinates(INIT_TIME_START + INIT_TIME_FREQUENCY),
        "ensemble_member": np.arange(31),
        "lead_time": pd.timedelta_range("0h", "240h", freq="3h").union(
            pd.timedelta_range("246h", "840h", freq="6h")
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


ENSEMBLE_VAR_CHUNKS: dict[Dim, int] = {
    "init_time": 1,  # one forecast per chunk
    "ensemble_member": 31,  # all ensemble members in one chunk
    "lead_time": 181,  # all lead times in one chunk
    "latitude": 73,  # 10 chunks over 721 pixels
    "longitude": 72,  # 20 chunks over 1440 pixels
}
STATISTIC_VAR_CHUNKS: dict[Dim, int] = {
    "init_time": 1,  # one forecast per chunk
    "lead_time": 181,  # all lead times in one chunk
    # Use 4x larger spatial chunks for ensemble statistic vars because they don't have ensemble dimension
    "latitude": (73 * 2),  # 5 chunks over 721 pixels
    "longitude": (72 * 2),  # 10 chunks over 1440 pixels
}
assert ENSEMBLE_VAR_DIMS == tuple(ENSEMBLE_VAR_CHUNKS.keys())
ENSEMBLE_VAR_CHUNKS_ORDERED = tuple(
    ENSEMBLE_VAR_CHUNKS[dim] for dim in ENSEMBLE_VAR_DIMS
)
STATISTIC_VAR_CHUNKS_ORDERED = tuple(
    STATISTIC_VAR_CHUNKS[dim] for dim in STATISTIC_VAR_DIMS
)

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


BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE = zarr.codecs.BloscCodec(
    typesize=4,
    cname="zstd",
    clevel=3,
    shuffle="shuffle",  # byte shuffle
).to_dict()

BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE = zarr.codecs.BloscCodec(
    typesize=8,
    cname="zstd",
    clevel=3,
    shuffle="shuffle",  # byte shuffle
).to_dict()

GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT: Final[int] = 7
GEFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL: Final[Literal["no-rounding"]] = (
    "no-rounding"
)

ENCODING_FLOAT32_DEFAULT = Encoding(
    dtype="float32",
    fill_value=np.nan,
    chunks=ENSEMBLE_VAR_CHUNKS_ORDERED,
    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
)
ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT = Encoding(
    dtype="float32",
    fill_value=np.nan,
    chunks=ENSEMBLE_VAR_CHUNKS_ORDERED,
    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
)

# 00 UTC forecasts have a 35 day lead time, the rest go out 16 days.
EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR = pd.Series(
    {
        0: pd.Timedelta(hours=840),
        6: pd.Timedelta(hours=384),
        12: pd.Timedelta(hours=384),
        18: pd.Timedelta(hours=384),
    }
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
        ),
        attrs=CoordinateAttrs(
            units="seconds since 1970-01-01 00:00:00",
            statistics_approximate=StatisticsApproximate(
                min=INIT_TIME_START.isoformat(), max="Present"
            ),
        ),
    ),
    Coordinate(
        name="ensemble_member",
        encoding=Encoding(
            dtype="uint16",
            fill_value=-1,
            chunks=len(_dim_coords["ensemble_member"]),
        ),
        attrs=CoordinateAttrs(
            units="realization",
            statistics_approximate=StatisticsApproximate(
                min=int(_dim_coords["ensemble_member"].min()),
                max=int(_dim_coords["ensemble_member"].max()),
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
        ),
        attrs=CoordinateAttrs(
            units="seconds since 1970-01-01 00:00:00",
            statistics_approximate=StatisticsApproximate(
                min=INIT_TIME_START.isoformat(), max="Present + 35 days"
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
            chunks=(INIT_TIME_COORDINATE_CHUNK_SIZE, len(_dim_coords["lead_time"])),
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


_DATA_VARIABLES = (
    # GEFSDataVar(
    #     name="visibility_surface",
    #     encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=15_000.0),
    #     attrs=DataVarAttrs(
    #         short_name="vis",
    #         long_name="Visibility",
    #         units="m",
    #         step_type="instant",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="VIS",
    #         grib_description='0[-] SFC="Ground or water surface"',
    #         grib_index_level="surface",
    #         gefs_file_type="s+b",
    #         index_position=1,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    # GEFSDataVar(
    #     name="wind_gust_surface",
    #     encoding=ENCODING_FLOAT32_DEFAULT,
    #     attrs=DataVarAttrs(
    #         short_name="gust",
    #         long_name="Wind speed (gust)",
    #         units="m/s",
    #         step_type="instant",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="GUST",
    #         grib_description='0[-] SFC="Ground or water surface"',
    #         grib_index_level="surface",
    #         gefs_file_type="s+b",
    #         index_position=2,
    #         keep_mantissa_bits=6,
    #     ),
    # ),
    # GEFSDataVar(
    #     name="pressure_mean_sea_level",
    #     encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=101_000.0),
    #     attrs=DataVarAttrs(
    #         short_name="mslet",
    #         long_name="MSLP (Eta model reduction)",
    #         units="Pa",
    #         step_type="instant",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="MSLET",
    #         grib_description='0[-] MSL="Mean sea level"',
    #         grib_index_level="mean sea level",
    #         gefs_file_type="s+b",
    #         index_position=3,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    GEFSDataVar(
        name="pressure_surface",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=100_000.0),
        attrs=DataVarAttrs(
            short_name="sp",
            long_name="Surface pressure",
            units="Pa",
            step_type="instant",
            standard_name="surface_air_pressure",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="PRES",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            gefs_file_type="s+a",
            index_position=4,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    # GEFSDataVar(
    #     name="soil_temperature_surface",
    #     encoding=ENCODING_FLOAT32_DEFAULT,
    #     attrs=DataVarAttrs(
    #         short_name="st",
    #         long_name="Soil temperature",
    #         units="C",
    #         step_type="instant",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="TSOIL",
    #         grib_description='0-0.1[m] DBLL="Depth below land surface"',
    #         grib_index_level="0-0.1 m below ground",
    #         gefs_file_type="s+a",
    #         index_position=5,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    # GEFSDataVar(
    #     name="soil_moisture_surface",
    #     encoding=ENCODING_FLOAT32_DEFAULT,
    #     attrs=DataVarAttrs(
    #         short_name="soilw",
    #         long_name="Volumetric soil moisture content",
    #         units="Fraction",
    #         step_type="instant",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="SOILW",
    #         grib_description='0-0.1[m] DBLL="Depth below land surface"',
    #         grib_index_level="0-0.1 m below ground",
    #         gefs_file_type="s+a",
    #         index_position=6,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    # GEFSDataVar(
    #     name="snow_water_equivalent_surface",
    #     encoding=ENCODING_FLOAT32_DEFAULT,
    #     attrs=DataVarAttrs(
    #         short_name="sdwe",
    #         long_name="Water equivalent of accumulated snow depth (deprecated)",
    #         units="kg/(m^2)",
    #         step_type="instant",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="WEASD",
    #         grib_description='0[-] SFC="Ground or water surface"',
    #         grib_index_level="surface",
    #         gefs_file_type="s+a",
    #         index_position=7,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    # GEFSDataVar(
    #     name="snow_depth_surface",
    #     encoding=ENCODING_FLOAT32_DEFAULT,
    #     attrs=DataVarAttrs(
    #         short_name="sde",
    #         long_name="Snow depth",
    #         units="m",
    #         step_type="instant",
    #         standard_name="lwe_thickness_of_surface_snow_amount",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="SNOD",
    #         grib_description='0[-] SFC="Ground or water surface"',
    #         grib_index_level="surface",
    #         gefs_file_type="s+a",
    #         index_position=8,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    # GEFSDataVar(
    #     name="sea_ice_thickness_surface",
    #     encoding=ENCODING_FLOAT32_DEFAULT,
    #     attrs=DataVarAttrs(
    #         short_name="sithick",
    #         long_name="Sea ice thickness",
    #         units="m",
    #         step_type="instant",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="ICETK",
    #         grib_description='0[-] SFC="Ground or water surface"',
    #         grib_index_level="surface",
    #         gefs_file_type="s+a",
    #         index_position=9,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    GEFSDataVar(
        name="temperature_2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="t2m",
            long_name="2 metre temperature",
            units="C",
            step_type="instant",
            standard_name="air_temperature",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="TMP",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            gefs_file_type="s+a",
            index_position=10,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    # GEFSDataVar(
    #     name="dew_point_2m",
    #     encoding=ENCODING_FLOAT32_DEFAULT,
    #     attrs=DataVarAttrs(
    #         short_name="d2m",
    #         long_name="2 metre dewpoint temperature",
    #         units="C",
    #         step_type="instant",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="DPT",
    #         grib_description='2[m] HTGL="Specified height level above ground"',
    #         grib_index_level="2 m above ground",
    #         gefs_file_type="s+b",
    #         index_position=11,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    GEFSDataVar(
        name="relative_humidity_2m",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=50.0),
        attrs=DataVarAttrs(
            short_name="r2",
            long_name="2 metre relative humidity",
            units="%",
            step_type="instant",
            standard_name="relative_humidity",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="RH",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            gefs_file_type="s+a",
            index_position=12,
            keep_mantissa_bits=6,
        ),
    ),
    GEFSDataVar(
        name="maximum_temperature_2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="tmax",
            long_name="Maximum temperature",
            units="C",
            step_type="max",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="TMAX",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            gefs_file_type="s+a",
            index_position=13,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    GEFSDataVar(
        name="minimum_temperature_2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="tmin",
            long_name="Minimum temperature",
            units="C",
            step_type="min",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="TMIN",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            gefs_file_type="s+a",
            index_position=14,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    GEFSDataVar(
        name="wind_u_10m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="u10",
            long_name="10 metre U wind component",
            units="m/s",
            step_type="instant",
            standard_name="eastward_wind",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="UGRD",
            grib_description='10[m] HTGL="Specified height level above ground"',
            grib_index_level="10 m above ground",
            gefs_file_type="s+a",
            index_position=15,
            keep_mantissa_bits=6,
        ),
    ),
    GEFSDataVar(
        name="wind_v_10m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="v10",
            long_name="10 metre V wind component",
            units="m/s",
            step_type="instant",
            standard_name="northward_wind",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="VGRD",
            grib_description='10[m] HTGL="Specified height level above ground"',
            grib_index_level="10 m above ground",
            gefs_file_type="s+a",
            index_position=16,
            keep_mantissa_bits=6,
        ),
    ),
    GEFSDataVar(
        name="wind_u_100m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="u100",
            long_name="100 metre U wind component",
            standard_name="eastward_wind",
            units="m/s",
            step_type="instant",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="UGRD",
            grib_description='100[m] HTGL="Specified height level above ground"',
            grib_index_level="100 m above ground",
            gefs_file_type="b",
            index_position=357,
            keep_mantissa_bits=6,
        ),
    ),
    GEFSDataVar(
        name="wind_v_100m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="v100",
            long_name="100 metre V wind component",
            units="m/s",
            step_type="instant",
            standard_name="northward_wind",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="VGRD",
            grib_description='100[m] HTGL="Specified height level above ground"',
            grib_index_level="100 m above ground",
            gefs_file_type="b",
            index_position=358,
            keep_mantissa_bits=6,
        ),
    ),
    GEFSDataVar(
        name="percent_frozen_precipitation_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="cpofp",
            long_name="Percent frozen precipitation",
            units="%",
            step_type="instant",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="CPOFP",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            gefs_file_type="s+b",
            index_position=17,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    GEFSDataVar(
        name="precipitation_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="tp",
            long_name="Total Precipitation",
            units="mm/s",
            step_type="accum",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="APCP",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            gefs_file_type="s+a",
            index_position=18,
            include_lead_time_suffix=True,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    GEFSDataVar(
        name="categorical_snow_surface",
        encoding=ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT,
        attrs=DataVarAttrs(
            short_name="csnow",
            long_name="Categorical snow",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="CSNOW",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            gefs_file_type="s+a",
            index_position=19,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL,
        ),
    ),
    GEFSDataVar(
        name="categorical_ice_pellets_surface",
        encoding=ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT,
        attrs=DataVarAttrs(
            short_name="cicep",
            long_name="Categorical ice pellets",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="CICEP",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            gefs_file_type="s+a",
            index_position=20,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL,
        ),
    ),
    GEFSDataVar(
        name="categorical_freezing_rain_surface",
        encoding=ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT,
        attrs=DataVarAttrs(
            short_name="cfrzr",
            long_name="Categorical freezing rain",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="CFRZR",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            gefs_file_type="s+a",
            index_position=21,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL,
        ),
    ),
    GEFSDataVar(
        name="categorical_rain_surface",
        encoding=ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT,
        attrs=DataVarAttrs(
            short_name="crain",
            long_name="Categorical rain",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="CRAIN",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            gefs_file_type="s+a",
            index_position=22,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_CATEGORICAL,
        ),
    ),
    # GEFSDataVar(
    #     name="mean_latent_heat_flux_surface",
    #     encoding=replace(ENCODING_FLOAT32_DEFAULT),
    #     attrs=DataVarAttrs(
    #         short_name="mslhf",
    #         long_name="Mean surface latent heat flux",
    #         units="W/(m^2)",
    #         step_type="avg",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="LHTFL",
    #         grib_description='0[-] SFC="Ground or water surface"',
    #         grib_index_level="surface",
    #         gefs_file_type="s+a",
    #         index_position=23,
    #         keep_mantissa_bits=6,
    #     ),
    # ),
    # GEFSDataVar(
    #     name="mean_sensible_heat_flux_surface",
    #     encoding=replace(ENCODING_FLOAT32_DEFAULT),
    #     attrs=DataVarAttrs(
    #         short_name="msshf",
    #         long_name="Mean surface sensible heat flux",
    #         units="W/(m^2)",
    #         step_type="avg",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="SHTFL",
    #         grib_description='0[-] SFC="Ground or water surface"',
    #         grib_index_level="surface",
    #         gefs_file_type="s+a",
    #         index_position=24,
    #         keep_mantissa_bits=6,
    #     ),
    # ),
    GEFSDataVar(
        name="precipitable_water_atmosphere",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="pwat",
            long_name="Precipitable water",
            units="kg/(m^2)",
            step_type="instant",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="PWAT",
            grib_description='0[-] EATM="Entire atmosphere (considered as a single layer)"',
            grib_index_level="entire atmosphere (considered as a single layer)",
            gefs_file_type="s+a",
            index_position=27,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    GEFSDataVar(
        name="total_cloud_cover_atmosphere",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=50.0),
        attrs=DataVarAttrs(
            short_name="tcc",
            long_name="Total Cloud Cover",
            units="%",
            step_type="avg",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="TCDC",
            grib_description='0[-] EATM="Entire Atmosphere"',
            grib_index_level="entire atmosphere",
            gefs_file_type="s+a",
            index_position=28,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    GEFSDataVar(
        name="geopotential_height_cloud_ceiling",
        encoding=replace(ENCODING_FLOAT32_DEFAULT),
        attrs=DataVarAttrs(
            short_name="gh",
            long_name="Geopotential height",
            units="gpm",
            step_type="instant",
            standard_name="geopotential_height",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="HGT",
            grib_description='0[-] CEIL="Cloud ceiling"',
            grib_index_level="cloud ceiling",
            gefs_file_type="s+b",
            index_position=29,
            keep_mantissa_bits=8,
        ),
    ),
    GEFSDataVar(
        name="downward_short_wave_radiation_flux_surface",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="sdswrf",
            long_name="Surface downward short-wave radiation flux",
            units="W/(m^2)",
            step_type="avg",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="DSWRF",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            gefs_file_type="s+a",
            index_position=30,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    GEFSDataVar(
        name="downward_long_wave_radiation_flux_surface",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=300.0),
        attrs=DataVarAttrs(
            short_name="sdlwrf",
            long_name="Surface downward long-wave radiation flux",
            units="W/(m^2)",
            step_type="avg",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="DLWRF",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            gefs_file_type="s+a",
            index_position=31,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
    # GEFSDataVar(
    #     name="upward_short_wave_radiation_flux_surface",
    #     encoding=ENCODING_FLOAT32_DEFAULT,
    #     attrs=DataVarAttrs(
    #         short_name="suswrf",
    #         long_name="Surface upward short-wave radiation flux",
    #         units="W/(m^2)",
    #         step_type="avg",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="USWRF",
    #         grib_description='0[-] SFC="Ground or water surface"',
    #         grib_index_level="surface",
    #         gefs_file_type="s+a",
    #         index_position=32,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    # GEFSDataVar(
    #     name="storm_relative_helicity_3000-0m",
    #     encoding=ENCODING_FLOAT32_DEFAULT,
    #     attrs=DataVarAttrs(
    #         short_name="hlcy",
    #         long_name="Storm relative helicity",
    #         units="m^2/s^2",  # TODO triple check our values really are in these units, gdal reports J/kg (equivalent?)
    #         step_type="instant",
    #     ),
    #     internal_attrs=InternalAttrs(
    #         grib_element="HLCY",
    #         grib_description='3000-0[m] HTGL="Specified height level above ground"',
    #         grib_index_level="3000-0 m above ground",
    #         gefs_file_type="s+b",
    #         index_position=35,
    #         keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
    #     ),
    # ),
    GEFSDataVar(
        name="pressure_reduced_to_mean_sea_level",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=101_000.0),
        attrs=DataVarAttrs(
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            step_type="instant",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="PRMSL",
            grib_description='0[-] MSL="Mean sea level"',
            grib_index_level="mean sea level",
            gefs_file_type="s+a",
            index_position=38,
            keep_mantissa_bits=GEFS_BITROUND_KEEP_MANTISSA_BITS_DEFAULT,
        ),
    ),
)

_STATISTIC_DATA_VARIABLES = tuple(
    replace(
        var,
        name=f"{var.name}_{ensemble_statistic}",
        attrs=replace(var.attrs, ensemble_statistic=ensemble_statistic),
        encoding=replace(var.encoding, chunks=STATISTIC_VAR_CHUNKS_ORDERED),
    )
    for var in _DATA_VARIABLES
    for ensemble_statistic in ["avg"]
    # Precomputed ensemble statistic files are not available for "b" files.
    # Should we compute the statistics ourselves?
    if var.internal_attrs.gefs_file_type not in ("b", "s+b")
)

DATA_VARIABLES: Sequence[GEFSDataVar] = _DATA_VARIABLES + _STATISTIC_DATA_VARIABLES
