from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
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

DATASET_ID = "noaa-gefs-forecast-35-day"
DATASET_VERSION = "0.2.0"

INIT_TIME_START = pd.Timestamp("2020-10-01T00:00")
INIT_TIME_FREQUENCY = pd.Timedelta("24h")

DATASET_ATTRIBUTES = DatasetAttributes(
    dataset_id=DATASET_ID,
    dataset_version=DATASET_VERSION,
    name="NOAA GEFS forecast, 35 day",
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
type Dim =            Literal["init_time", "ensemble_member", "lead_time", "latitude", "longitude"]  # fmt: off
VAR_DIMS: tuple[Dim, ... ] = ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")  # fmt: off

APPEND_DIMENSION = "init_time"


def get_template_dimension_coordinates() -> dict[str, Any]:
    return {
        "init_time": get_init_time_coordinates(INIT_TIME_START + INIT_TIME_FREQUENCY),
        "ensemble_member": np.arange(31),
        "lead_time": pd.timedelta_range("0h", "240h", freq="3h").union(
            pd.timedelta_range("246h", "840h", freq="6h")
        ),
        **get_shared_template_dimension_coordinates(),
    }


def get_init_time_coordinates(
    init_time_end: DatetimeLike,
) -> pd.DatetimeIndex:
    return pd.date_range(
        INIT_TIME_START, init_time_end, freq=INIT_TIME_FREQUENCY, inclusive="left"
    )


# CHUNKS
# These chunks are about 2mb of uncompressed float32s
VAR_CHUNKS: dict[Dim, int] = {
    "init_time": 1,  # one forecast per chunk
    "ensemble_member": 31,  # all ensemble members in one chunk
    "lead_time": 64,  # 3 chunks, first chunk includes days 0-7, second days 8-mid day 21, third days 21-35
    "latitude": 17,  # 43 chunks over 721 pixels
    "longitude": 16,  # 90 chunks over 1440 pixels
}
# SHARDS
# About 300-550MB compressed, about 3GB uncompressed
VAR_SHARDS: dict[Dim, int] = {
    "init_time": 1,  # one forecast per shard
    "ensemble_member": 31,  # all ensemble members in one shard
    "lead_time": VAR_CHUNKS["lead_time"] * 3,  # all lead times in one shard
    "latitude": VAR_CHUNKS["latitude"] * 22,  # 2 shards over 721 pixels
    "longitude": VAR_CHUNKS["longitude"] * 23,  # 4 shards over 1440 pixels
}
assert VAR_DIMS == tuple(VAR_CHUNKS.keys())
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
    *get_shared_coordinate_configs(),
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
        name="ensemble_member",
        encoding=Encoding(
            dtype="uint16",
            fill_value=-1,
            chunks=len(_dim_coords["ensemble_member"]),
            shards=len(_dim_coords["ensemble_member"]),
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
            shards=(INIT_TIME_COORDINATE_CHUNK_SIZE, len(_dim_coords["lead_time"])),
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
)

DATA_VARIABLES: Sequence[GEFSDataVar] = get_shared_data_var_configs(
    VAR_CHUNKS_ORDERED, VAR_SHARDS_ORDERED
)
