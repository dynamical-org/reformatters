from typing import Any, Literal

import numpy as np
import pandas as pd
from numcodecs import BitRound, Blosc, Delta  # type: ignore

from common.types import DatetimeLike

from .config_models import Coordinate, DataVar, DataVarAttrs, Encoding, InternalAttrs

DATASET_ID = "noaa-gefs-forecast"

# Silly to define this twice, but typing.get_args() doesn't guarantee the return order,
# type parameters can't be constants and the order in DIMS is important so here we are.
type Dim =        Literal["init_time", "ensemble_member", "lead_time", "latitude", "longitude"]  # fmt: off
DIMS: tuple[Dim, ... ] = ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")  # fmt: off

INIT_TIME_START = pd.Timestamp("2024-09-01T00:00")
INIT_TIME_FREQUENCY = pd.Timedelta("6h")


def get_template_coordinates() -> dict[Dim, Any]:
    return {
        "init_time": get_init_time_coordinates(INIT_TIME_START + INIT_TIME_FREQUENCY),
        "ensemble_member": np.arange(31),
        "lead_time": pd.timedelta_range("0h", "240h", freq="3h"),
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

COORDINATES: tuple[Coordinate, ...] = (
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


DATA_VARIABLES: tuple[DataVar, ...] = (
    DataVar(
        name="t2m",
        encoding=ENCODING_FLOAT32_DEFAULT,
        attrs=DataVarAttrs(
            long_name="2 metre temperature",
            standard_name="air_temperature",
            units="C",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="TMP",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            noaa_file_type="s+a",
        ),
    ),
    DataVar(
        name="u10",
        encoding=ENCODING_FLOAT32_DEFAULT.model_copy(
            update={"filters": [BitRound(keepbits=6)]}
        ),
        attrs=DataVarAttrs(
            long_name="10 metre U wind component",
            standard_name="eastward_wind",
            units="m s**-1",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="UGRD",
            grib_description='10[m] HTGL="Specified height level above ground"',
            grib_index_level="10 m above ground",
            noaa_file_type="s+a",
        ),
    ),
    DataVar(
        name="u100",
        encoding=ENCODING_FLOAT32_DEFAULT.model_copy(
            update={"filters": [BitRound(keepbits=6)]}
        ),
        attrs=DataVarAttrs(
            long_name="100 metre U wind component",
            standard_name="eastward_wind",
            units="m s**-1",
            step_type="instant",
        ),
        internal_attrs=InternalAttrs(
            grib_element="UGRD",
            grib_description='100[m] HTGL="Specified height level above ground"',
            grib_index_level="100 m above ground",
            noaa_file_type="b",
        ),
    ),
)

ENCODING: dict[str, Encoding] = {
    **{coord.name: coord.encoding for coord in COORDINATES},
    **{var.name: var.encoding for var in DATA_VARIABLES},
}
assert len(ENCODING) == len(COORDINATES) + len(DATA_VARIABLES)

# _ENCODING = {
#     "init_time": {
#         "dtype": np.int64,
#         "filters": [Delta(np.int64)],
#         "compressor": Blosc(cname="zstd"),
#         "calendar": "proleptic_gregorian",
#         "units": "seconds since 1970-01-01 00:00:00",
#         "chunks": -1,
#     },
#     "ensemble_member": {
#         "dtype": np.uint16,
#         "chunks": -1,
#     },
#     "lead_time": {
#         "dtype": np.int64,
#         "compressor": Blosc(cname="zstd"),
#         "units": "seconds",
#         "chunks": -1,
#     },
#     "latitude": {
#         "dtype": np.float64,
#         "compressor": Blosc(cname="zstd"),
#         "chunks": -1,
#     },
#     "longitude": {
#         "dtype": np.float64,
#         "compressor": Blosc(cname="zstd"),
#         "chunks": -1,
#     },
#     "valid_time": {
#         "dtype": np.int64,
#         "filters": [Delta(np.int64)],
#         "compressor": Blosc(cname="zstd"),
#         "calendar": "proleptic_gregorian",
#         "units": "seconds since 1970-01-01 00:00:00",
#         "chunks": [-1, -1],
#     },
#     "cfrzr": _CATEGORICAL_WITH_MISSING_DEFAULT,
#     "cicep": _CATEGORICAL_WITH_MISSING_DEFAULT,
#     "cpofp": _FLOAT_DEFAULT,
#     "crain": _CATEGORICAL_WITH_MISSING_DEFAULT,
#     "csnow": _CATEGORICAL_WITH_MISSING_DEFAULT,
#     "d2m": {**_FLOAT_DEFAULT, "add_offset": 273.15},
#     "gh": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=8)]},
#     "gust": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
#     "hlcy": _FLOAT_DEFAULT,
#     "mslet": {**_FLOAT_DEFAULT, "add_offset": 101_000.0},
#     "mslhf": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
#     "msshf": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
#     "prmsl": {**_FLOAT_DEFAULT, "add_offset": 101_000.0},
#     "pwat": _FLOAT_DEFAULT,
#     "r2": {**_FLOAT_DEFAULT, "add_offset": 50.0, "filters": [BitRound(keepbits=6)]},
#     "sde": _FLOAT_DEFAULT,
#     "sdlwrf": {**_FLOAT_DEFAULT, "add_offset": 300.0},
#     "sdswrf": _FLOAT_DEFAULT,
#     "sdwe": _FLOAT_DEFAULT,
#     "sithick": _FLOAT_DEFAULT,
#     "soilw": _FLOAT_DEFAULT,
#     "sp": {**_FLOAT_DEFAULT, "add_offset": 100_000.0},
#     "st": {**_FLOAT_DEFAULT, "add_offset": 273.15},
#     "suswrf": _FLOAT_DEFAULT,
#     # "t2m": {**_FLOAT_DEFAULT, "add_offset": 273.15},
#     "t2m": Encoding(
#         dtype=np.float32,
#         chunks=_CHUNKS_ORDERED,
#         filters=[BitRound(keepbits=7)],
#         compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
#     ),
#     "tcc": {**_FLOAT_DEFAULT, "add_offset": 50.0},
#     "tmax": {**_FLOAT_DEFAULT, "add_offset": 273.15},
#     "tmin": {**_FLOAT_DEFAULT, "add_offset": 273.15},
#     "tp": _FLOAT_DEFAULT,
#     "u10": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
#     # "u100": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
#     "v10": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
#     # "v100": {**_FLOAT_DEFAULT, "filters": [BitRound(keepbits=6)]},
#     "vis": {**_FLOAT_DEFAULT, "add_offset": 15_000.0},
# }

# # TODO: Some of the s+a should actually be s+b, check the
# # index files to see which one.
# _CUSTOM_ATTRIBUTES = {
#     "cfrzr": {
#         "noaa_file_type": "s+a",
#         "grib_element": "CFRZR",
#         "index_level": "surface",
#     },
#     "cicep": {"noaa_file_type": "s+a"},
#     "cpofp": {"noaa_file_type": "s+a"},
#     "crain": {"noaa_file_type": "s+a"},
#     "csnow": {"noaa_file_type": "s+a"},
#     "d2m": {"noaa_file_type": "s+a"},
#     "gh": {"noaa_file_type": "s+a"},
#     "gust": {"noaa_file_type": "s+a"},
#     "hlcy": {"noaa_file_type": "s+a"},
#     "mslet": {"noaa_file_type": "s+a"},
#     "mslhf": {"noaa_file_type": "s+a"},
#     "msshf": {"noaa_file_type": "s+a"},
#     "prmsl": {
#         "noaa_file_type": "s+a",
#         "grib_element": "PRMSL",
#         "index_level": "mean sea level",
#     },
#     "pwat": {"noaa_file_type": "s+a"},
#     "r2": {"noaa_file_type": "s+a"},
#     "sde": {"noaa_file_type": "s+a"},
#     "sdlwrf": {"noaa_file_type": "s+a"},
#     "sdswrf": {"noaa_file_type": "s+a"},
#     "sdwe": {"noaa_file_type": "s+a"},
#     "sithick": {"noaa_file_type": "s+a"},
#     "soilw": {"noaa_file_type": "s+a"},
#     "sp": {"noaa_file_type": "s+a"},
#     "st": {"noaa_file_type": "s+a"},
#     "suswrf": {"noaa_file_type": "s+a"},
#     "t2m": {
#         "noaa_file_type": "s+a",
#         "index_level": "2 m above ground",
#         "grib_element": "TMP",
#         "grib_description": '2[m] HTGL="Specified height level above ground"',
#     },
#     "tcc": {"noaa_file_type": "s+a"},
#     "tmax": {"noaa_file_type": "s+a"},
#     "tmin": {"noaa_file_type": "s+a"},
#     "tp": {"noaa_file_type": "s+a"},
#     "u10": {
#         "noaa_file_type": "s+a",
#         "index_level": "10 m above ground",
#         "grib_element": "UGRD",
#         "grib_description": '10[m] HTGL="Specified height level above ground"',
#     },
#     "u100": {
#         "noaa_file_type": "b",
#         "grib_element": "UGRD",
#         "index_level": "100 m above ground",
#     },
#     "v10": {
#         "noaa_file_type": "s+a",
#         "grib_element": "VGRD",
#         "index_level": "10 m above ground",
#     },
#     "v100": {
#         "noaa_file_type": "b",
#         "grib_element": "VGRD",
#         "index_level": "100 m above ground",
#     },
#     "vis": {"noaa_file_type": "s+b", "grib_element": "VIS", "index_level": "surface"},
# }
