from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numcodecs import BitRound, Blosc, Delta  # type: ignore

from common.types import DatetimeLike

from .config_models import (
    Coordinate,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    InternalAttrs,
    replace,
)

DATASET_ATTRIBUTES = DatasetAttributes(
    dataset_id="noaa-gefs-forecast",
    name="NOAA GEFS forecast",
    description="Weather forecasts from the Global Ensemble Forecast System (GEFS) operated by NOAA NWS NCEP.",
    attribution="NOAA NWS NCEP GEFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
)

# Silly to list dims twice, but typing.get_args() doesn't guarantee the return order,
# the order in DIMS is important, and type parameters can't be constants.
type Dim =        Literal["init_time", "ensemble_member", "lead_time", "latitude", "longitude"]  # fmt: off
DIMS: tuple[Dim, ... ] = ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")  # fmt: off

INIT_TIME_START = pd.Timestamp("2024-09-01T00:00")
INIT_TIME_FREQUENCY = pd.Timedelta("24h")


def get_template_dimension_coordinates() -> dict[Dim, Any]:
    return {
        "init_time": get_init_time_coordinates(INIT_TIME_START + INIT_TIME_FREQUENCY),
        "ensemble_member": np.arange(31),
        "lead_time": pd.timedelta_range("0h", "840h", freq="3h"),
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


CHUNKS: dict[Dim, int] = {
    "init_time": 1,  # one forecast per chunk
    "ensemble_member": 31,  # all ensemble members in one chunk
    "lead_time": 125,  # all lead times in one chunk
    "latitude": 73,  # 10 chunks over 721 pixels
    "longitude": 72,  # 20 chunks over 1440 pixels
}
assert DIMS == tuple(CHUNKS.keys())
CHUNKS_ORDERED = tuple(CHUNKS[dim] for dim in DIMS)

ENCODING_FLOAT32_DEFAULT = Encoding(
    dtype="float32",
    chunks=CHUNKS_ORDERED,
    filters=[BitRound(keepbits=7)],
    compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
)
ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT = Encoding(
    dtype="float32",
    chunks=CHUNKS_ORDERED,
    compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
)

COORDINATES: Sequence[Coordinate] = (
    Coordinate(
        name="init_time",
        encoding=Encoding(
            dtype="int64",
            filters=[Delta("int64")],
            compressor=Blosc(cname="zstd"),
            calendar="proleptic_gregorian",
            units="seconds since 1970-01-01 00:00:00",
            chunks=-1,
        ),
    ),
    Coordinate(
        name="ensemble_member",
        encoding=Encoding(
            dtype="uint16",
            chunks=-1,
        ),
    ),
    Coordinate(
        name="lead_time",
        encoding=Encoding(
            dtype="int64",
            compressor=Blosc(cname="zstd"),
            units="seconds",
            chunks=-1,
        ),
    ),
    Coordinate(
        name="latitude",
        encoding=Encoding(
            dtype="float64",
            compressor=Blosc(cname="zstd"),
            chunks=-1,
        ),
    ),
    Coordinate(
        name="longitude",
        encoding=Encoding(
            dtype="float64",
            compressor=Blosc(cname="zstd"),
            chunks=-1,
        ),
    ),
    Coordinate(
        name="valid_time",
        encoding=Encoding(
            dtype="int64",
            filters=[Delta("int64")],
            compressor=Blosc(cname="zstd"),
            calendar="proleptic_gregorian",
            units="seconds since 1970-01-01 00:00:00",
            chunks=(-1, -1),
        ),
    ),
)


DATA_VARIABLES: Sequence[DataVar] = (
    DataVar(
        name="vis",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=15_000.0),
        attrs=DataVarAttrs(
            short_name="vis",
            long_name="Visibility",
            units="m",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="VIS",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+b",
            index_position=1,
        ),
    ),
    DataVar(
        name="gust",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, filters=[BitRound(keepbits=6)]),
        attrs=DataVarAttrs(
            short_name="gust",
            long_name="Wind speed (gust)",
            units="m/s",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="GUST",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+b",
            index_position=2,
        ),
    ),
    DataVar(
        name="mslet",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=101_000.0),
        attrs=DataVarAttrs(
            short_name="mslet",
            long_name="MSLP (Eta model reduction)",
            units="Pa",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="MSLET",
            grib_description='0[-] MSL="Mean sea level"',
            grib_index_level="mean sea level",
            noaa_file_type="s+b",
            index_position=3,
        ),
    ),
    DataVar(
        name="sp",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=100_000.0),
        attrs=DataVarAttrs(
            short_name="sp",
            long_name="Surface pressure",
            units="Pa",
            step_type="instant",
            standard_name="surface_air_pressure",
        ),
        internal_attrs=InternalAttrs(
            grib_element="PRES",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=4,
        ),
    ),
    DataVar(
        name="st",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="st",
            long_name="Soil temperature",
            units="C",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="TSOIL",
            grib_description='0-0.1[m] DBLL="Depth below land surface"',
            grib_index_level="0-0.1 m below ground",
            noaa_file_type="s+a",
            index_position=5,
        ),
    ),
    DataVar(
        name="soilw",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="soilw",
            long_name="Volumetric soil moisture content",
            units="Fraction",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="SOILW",
            grib_description='0-0.1[m] DBLL="Depth below land surface"',
            grib_index_level="0-0.1 m below ground",
            noaa_file_type="s+a",
            index_position=6,
        ),
    ),
    DataVar(
        name="sdwe",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="sdwe",
            long_name="Water equivalent of accumulated snow depth (deprecated)",
            units="kg/(m^2)",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="WEASD",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=7,
        ),
    ),
    DataVar(
        name="sde",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="sde",
            long_name="Snow depth",
            units="m",
            step_type="instant",
            standard_name="lwe_thickness_of_surface_snow_amount",
        ),
        internal_attrs=InternalAttrs(
            grib_element="SNOD",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=8,
        ),
    ),
    DataVar(
        name="sithick",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="sithick",
            long_name="Sea ice thickness",
            units="m",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="ICETK",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=9,
        ),
    ),
    DataVar(
        name="t2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="t2m",
            long_name="2 metre temperature",
            units="C",
            step_type="instant",
            standard_name="air_temperature",
        ),
        internal_attrs=InternalAttrs(
            grib_element="TMP",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            noaa_file_type="s+a",
            index_position=10,
        ),
    ),
    DataVar(
        name="d2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="d2m",
            long_name="2 metre dewpoint temperature",
            units="C",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="DPT",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            noaa_file_type="s+b",
            index_position=11,
        ),
    ),
    DataVar(
        name="r2",
        encoding=replace(
            ENCODING_FLOAT32_DEFAULT, add_offset=50.0, filters=[BitRound(keepbits=6)]
        ),
        attrs=DataVarAttrs(
            short_name="r2",
            long_name="2 metre relative humidity",
            units="%",
            step_type="instant",
            standard_name="relative_humidity",
        ),
        internal_attrs=InternalAttrs(
            grib_element="RH",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            noaa_file_type="s+a",
            index_position=12,
        ),
    ),
    DataVar(
        name="tmax",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="tmax",
            long_name="Maximum temperature",
            units="C",
            step_type="max",
        ),
        internal_attrs=InternalAttrs(
            grib_element="TMAX",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            noaa_file_type="s+a",
            index_position=13,
        ),
    ),
    DataVar(
        name="tmin",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="tmin",
            long_name="Minimum temperature",
            units="C",
            step_type="min",
        ),
        internal_attrs=InternalAttrs(
            grib_element="TMIN",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            noaa_file_type="s+a",
            index_position=14,
        ),
    ),
    DataVar(
        name="u10",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, filters=[BitRound(keepbits=6)]),
        attrs=DataVarAttrs(
            short_name="u10",
            long_name="10 metre U wind component",
            units="m/s",
            step_type="instant",
            standard_name="eastward_wind",
        ),
        internal_attrs=InternalAttrs(
            grib_element="UGRD",
            grib_description='10[m] HTGL="Specified height level above ground"',
            grib_index_level="10 m above ground",
            noaa_file_type="s+a",
            index_position=15,
        ),
    ),
    DataVar(
        name="v10",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, filters=[BitRound(keepbits=6)]),
        attrs=DataVarAttrs(
            short_name="v10",
            long_name="10 metre V wind component",
            units="m/s",
            step_type="instant",
            standard_name="northward_wind",
        ),
        internal_attrs=InternalAttrs(
            grib_element="VGRD",
            grib_description='10[m] HTGL="Specified height level above ground"',
            grib_index_level="10 m above ground",
            noaa_file_type="s+a",
            index_position=16,
        ),
    ),
    DataVar(
        name="u100",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, filters=[BitRound(keepbits=6)]),
        attrs=DataVarAttrs(
            short_name="u100",
            long_name="100 metre U wind component",
            standard_name="eastward_wind",
            units="m/s",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="UGRD",
            grib_description='100[m] HTGL="Specified height level above ground"',
            grib_index_level="100 m above ground",
            noaa_file_type="b",
            index_position=357,
        ),
    ),
    DataVar(
        name="v100",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, filters=[BitRound(keepbits=6)]),
        attrs=DataVarAttrs(
            short_name="v100",
            long_name="100 metre V wind component",
            units="m/s",
            step_type="instant",
            standard_name="northward_wind",
        ),
        internal_attrs=InternalAttrs(
            grib_element="VGRD",
            grib_description='100[m] HTGL="Specified height level above ground"',
            grib_index_level="100 m above ground",
            noaa_file_type="b",
            index_position=358,
        ),
    ),
    DataVar(
        name="cpofp",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="cpofp",
            long_name="Percent frozen precipitation",
            units="%",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="CPOFP",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+b",
            index_position=17,
        ),
    ),
    DataVar(
        name="tp",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="tp",
            long_name="Total Precipitation",
            units="kg/(m^2)",
            step_type="accum",
        ),
        internal_attrs=InternalAttrs(
            grib_element="APCP",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=18,
            include_lead_time_suffix=True,
        ),
    ),
    DataVar(
        name="csnow",
        encoding=ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT,
        attrs=DataVarAttrs(
            short_name="csnow",
            long_name="Categorical snow",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="CSNOW",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=19,
        ),
    ),
    DataVar(
        name="cicep",
        encoding=ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT,
        attrs=DataVarAttrs(
            short_name="cicep",
            long_name="Categorical ice pellets",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="CICEP",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=20,
        ),
    ),
    DataVar(
        name="cfrzr",
        encoding=ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT,
        attrs=DataVarAttrs(
            short_name="cfrzr",
            long_name="Categorical freezing rain",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="CFRZR",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=21,
        ),
    ),
    DataVar(
        name="crain",
        encoding=ENCODING_CATEGORICAL_WITH_MISSING_DEFAULT,
        attrs=DataVarAttrs(
            short_name="crain",
            long_name="Categorical rain",
            units="0=no; 1=yes",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="CRAIN",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=22,
        ),
    ),
    DataVar(
        name="mslhf",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, filters=[BitRound(keepbits=6)]),
        attrs=DataVarAttrs(
            short_name="mslhf",
            long_name="Mean surface latent heat flux",
            units="W/(m^2)",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="LHTFL",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=23,
        ),
    ),
    DataVar(
        name="msshf",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, filters=[BitRound(keepbits=6)]),
        attrs=DataVarAttrs(
            short_name="msshf",
            long_name="Mean surface sensible heat flux",
            units="W/(m^2)",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="SHTFL",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=24,
        ),
    ),
    DataVar(
        name="pwat",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="pwat",
            long_name="Precipitable water",
            units="kg/(m^2)",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="PWAT",
            grib_description='0[-] EATM="Entire atmosphere (considered as a single layer)"',
            grib_index_level="entire atmosphere (considered as a single layer)",
            noaa_file_type="s+a",
            index_position=27,
        ),
    ),
    DataVar(
        name="tcc",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=50.0),
        attrs=DataVarAttrs(
            short_name="tcc",
            long_name="Total Cloud Cover",
            units="%",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="TCDC",
            grib_description='0[-] EATM="Entire Atmosphere"',
            grib_index_level="entire atmosphere",
            noaa_file_type="s+a",
            index_position=28,
        ),
    ),
    DataVar(
        name="gh",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, filters=[BitRound(keepbits=8)]),
        attrs=DataVarAttrs(
            short_name="gh",
            long_name="Geopotential height",
            units="gpm",
            step_type="instant",
            standard_name="geopotential_height",
        ),
        internal_attrs=InternalAttrs(
            grib_element="HGT",
            grib_description='0[-] CEIL="Cloud ceiling"',
            grib_index_level="cloud ceiling",
            noaa_file_type="s+b",
            index_position=29,
        ),
    ),
    DataVar(
        name="sdswrf",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="sdswrf",
            long_name="Surface downward short-wave radiation flux",
            units="W/(m^2)",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="DSWRF",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=30,
        ),
    ),
    DataVar(
        name="sdlwrf",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=300.0),
        attrs=DataVarAttrs(
            short_name="sdlwrf",
            long_name="Surface downward long-wave radiation flux",
            units="W/(m^2)",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="DLWRF",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=31,
        ),
    ),
    DataVar(
        name="suswrf",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="suswrf",
            long_name="Surface upward short-wave radiation flux",
            units="W/(m^2)",
            step_type="avg",
        ),
        internal_attrs=InternalAttrs(
            grib_element="USWRF",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_index_level="surface",
            noaa_file_type="s+a",
            index_position=32,
        ),
    ),
    DataVar(
        name="hlcy",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            short_name="hlcy",
            long_name="Storm relative helicity",
            units="m^2/s^2",  # TODO triple check our values really are in these units, gdal reports J/kg (equivalent?)
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="HLCY",
            grib_description='3000-0[m] HTGL="Specified height level above ground"',
            grib_index_level="3000-0 m above ground",
            noaa_file_type="s+b",
            index_position=35,
        ),
    ),
    DataVar(
        name="prmsl",
        encoding=replace(ENCODING_FLOAT32_DEFAULT, add_offset=101_000.0),
        attrs=DataVarAttrs(
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="PRMSL",
            grib_description='0[-] MSL="Mean sea level"',
            grib_index_level="mean sea level",
            noaa_file_type="s+a",
            index_position=38,
        ),
    ),
)
