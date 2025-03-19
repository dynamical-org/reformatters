from collections.abc import Sequence
from typing import Any, Final, Literal

import pandas as pd

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.types import DatetimeLike
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.gefs.common_gefs_template_config import (
    get_shared_coordinate_configs,
    get_shared_data_var_configs,
    get_shared_template_dimension_coordinates,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

DATASET_ID = "noaa-gefs-analysis"
DATASET_VERSION = "0.0.1"

TIME_START = pd.Timestamp("2020-10-01T00:00")
TIME_FREQUENCY = pd.Timedelta("3h")

DATASET_ATTRIBUTES = DatasetAttributes(
    dataset_id=DATASET_ID,
    dataset_version=DATASET_VERSION,
    name="NOAA GEFS analysis",
    description="Weather analysis from the Global Ensemble Forecast System (GEFS) operated by NOAA NWS NCEP.",
    attribution="NOAA NWS NCEP GEFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
    spatial_domain="Global",
    spatial_resolution="0.25 degrees (~20km)",
    time_domain=f"{TIME_START} UTC to Present",
    time_resolution=f"{TIME_FREQUENCY.total_seconds() / (60 * 60)} hours",
)

# Silly to list dims twice, but typing.get_args() doesn't guarantee the return order,
# the order in DIMS is important, and type parameters can't be constants.
type Dim =       Literal["time", "latitude", "longitude"]  # fmt: off
DIMS: tuple[Dim, ...] = ("time", "latitude", "longitude")  # fmt: off

APPEND_DIMENSION: Final[Dim] = "time"

ANALYSIS_ENSEMBLE_MEMBER = 0  # the GEFS control member


def get_template_dimension_coordinates() -> dict[str, Any]:
    return {
        "time": get_time_coordinates(TIME_START + TIME_FREQUENCY),
        **get_shared_template_dimension_coordinates(),
    }


def get_time_coordinates(
    time_end: DatetimeLike,
) -> pd.DatetimeIndex:
    return pd.date_range(TIME_START, time_end, freq=TIME_FREQUENCY, inclusive="left")


# CHUNKS
VAR_CHUNKS: dict[Dim, int] = {
    "time": 2 * 365 * (24 // 3),  # 2 years of 3 hourly data
    "latitude": 17,  # 43 chunks over 721 pixels
    "longitude": 16,  # 90 chunks over 1440 pixels
}

# SHARDS
VAR_SHARDS: dict[Dim, int] = {
    "time": VAR_CHUNKS["time"],
    "latitude": VAR_CHUNKS["latitude"] * 22,  # 2 shards over 721 pixels
    "longitude": VAR_CHUNKS["longitude"] * 23,  # 4 shards over 1440 pixels
}
assert DIMS == tuple(VAR_CHUNKS.keys())
VAR_CHUNKS_ORDERED = tuple(VAR_CHUNKS[dim] for dim in DIMS)
VAR_SHARDS_ORDERED = tuple(VAR_SHARDS[dim] for dim in DIMS)


# The time dimension is our append dimension during updates.
# We also want coordinates to be in a single chunk for dataset open speed.
# By fixing the chunk size for coordinates along the append dimension to
# something much larger than we will really use, the array is always
# a fixed underlying chunk size and values in it can be safely updated
# prior to metadata document updates that increase the reported array size.
# This is a zarr format hack to allow expanding an array safely and requires
# that new array values are written strictly before new metadata is written
# (doing this correctly is a key benefit of icechunk).
TIME_COORDINATE_CHUNK_SIZE = int(pd.Timedelta(days=365 * 50) / TIME_FREQUENCY)


COORDINATES: Sequence[Coordinate] = (
    *get_shared_coordinate_configs(),
    Coordinate(
        name="time",
        encoding=Encoding(
            dtype="int64",
            fill_value=0,
            compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
            calendar="proleptic_gregorian",
            units="seconds since 1970-01-01 00:00:00",
            chunks=TIME_COORDINATE_CHUNK_SIZE,
            shards=TIME_COORDINATE_CHUNK_SIZE,
        ),
        attrs=CoordinateAttrs(
            units="seconds since 1970-01-01 00:00:00",
            statistics_approximate=StatisticsApproximate(
                min=TIME_START.isoformat(), max="Present"
            ),
        ),
    ),
)


DATA_VARIABLES: Sequence[GEFSDataVar] = get_shared_data_var_configs(
    VAR_CHUNKS_ORDERED, VAR_SHARDS_ORDERED
)
