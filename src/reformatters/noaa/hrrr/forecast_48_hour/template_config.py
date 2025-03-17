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

DATASET_ID = "noaa-hrrr-forecast-48-hour"
DATASET_VERSION = "0.0.0"

INIT_TIME_START = pd.Timestamp("2014-07-30T18:00")
INIT_TIME_FREQUENCY = pd.Timedelta("6h")
LEAD_TIME_FREQUENCY = pd.Timedelta("1h")

DATASET_ATTRIBUTES = DatasetAttributes(
    dataset_id=DATASET_ID,
    dataset_version=DATASET_VERSION,
    name="NOAA HRRR forecast, 48 hour",
    description="Weather forecasts from the High Resolution Rapid Refresh (HRRR) model operated by NOAA NWS NCEP.",
    attribution="NOAA NWS NCEP HRRR data processed by dynamical.org from NOAA Open Data Dissemination archives.",
    spatial_domain="CONUS",
    spatial_resolution="3km",
    time_domain=f"Forecasts initialized {INIT_TIME_START} UTC to Present",
    time_resolution="Forecasts initialized every 6 hours.",
    forecast_domain="Forecast lead time 0-48 hours ahead",
    forecast_resolution="Hourly",
)

type Dim =       Literal["init_time", "lead_time", "latitude", "longitude"]  # fmt: off
DIMS: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")  # fmt: off


# TODO: figure out latitude/longitude dimensions
def get_template_dimension_coordinates() -> dict[str, Any]:
    raise NotImplementedError()
    # return {
    #     "init_time": get_init_time_coordinates(INIT_TIME_START + INIT_TIME_FREQUENCY),
    #     "lead_time": pd.timedelta_range("0h", "48h", freq=LEAD_TIME_FREQUENCY),
    #     "x": np.arange(1059),
    #     "y": np.arange(1799)
    # }


def get_init_time_coordinates(
    init_time_end: DatetimeLike,
) -> pd.DatetimeIndex:
    return pd.date_range(
        INIT_TIME_START, init_time_end, freq=INIT_TIME_FREQUENCY, inclusive="left"
    )


# TODO
# CHUNKS
# These chunks are about XXXmb of uncompressed float32s
CHUNKS: dict[Dim, int] = {}

# TODO
# SHARDS
# About XXXMB compressed, about XXGB uncompressed
SHARDS: dict[Dim, int] = {}

CHUNKS_ORDERED = tuple(CHUNKS[dim] for dim in DIMS)
SHARDS_ORDERED = tuple(SHARDS[dim] for dim in DIMS)


# TODO: review chunksize
# The init time dimension is our append dimension during updates.
# We also want coordinates to be in a single chunk for dataset open speed.
# By fixing the chunk size for coordinates along the append dimension to
# something much larger than we will really use, the array is always
# a fixed underlying chunk size and values in it can be safely updated
# prior to metadata document updates that increase the reported array size.
# This is a zarr format hack to allow expanding an array safely and requires
# that new array values are written strictly before new metadata is written
# (doing this correctly is a key benefit of icechunk).
INIT_TIME_COORDINATE_CHUNK_SIZE = int(
    pd.Timedelta(hours=365 * 24 * 15) / INIT_TIME_FREQUENCY
)

_dim_coords = get_template_dimension_coordinates()

# TODO
COORDINATES: Sequence[Coordinate] = (
    Coordinate(
        name="init_time",
        encoding=Encoding(
            dtype="int64",
            fill_value=0,
            # TODO
            # compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
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
            # TODO
            # compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
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
        name="x",
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            # TODO
            # compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
            chunks=len(_dim_coords["x"]),
            shards=len(_dim_coords["x"]),
        ),
        attrs=CoordinateAttrs(
            units="unitless",
            statistics_approximate=StatisticsApproximate(
                min=_dim_coords["x"].min(),
                max=_dim_coords["x"].max(),
            ),
        ),
    ),
    Coordinate(
        name="y",
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            # TODO
            # compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
            chunks=len(_dim_coords["y"]),
            shards=len(_dim_coords["y"]),
        ),
        attrs=CoordinateAttrs(
            units="unitless",
            statistics_approximate=StatisticsApproximate(
                min=_dim_coords["y"].min(),
                max=_dim_coords["y"].max(),
            ),
        ),
    ),
    Coordinate(
        name="valid_time",
        encoding=Encoding(
            dtype="int64",
            fill_value=0,
            # TODO
            # compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
            calendar="proleptic_gregorian",
            units="seconds since 1970-01-01 00:00:00",
            chunks=(INIT_TIME_COORDINATE_CHUNK_SIZE, len(_dim_coords["lead_time"])),
            shards=(INIT_TIME_COORDINATE_CHUNK_SIZE, len(_dim_coords["lead_time"])),
        ),
        attrs=CoordinateAttrs(
            units="seconds since 1970-01-01 00:00:00",
            statistics_approximate=StatisticsApproximate(
                min=INIT_TIME_START.isoformat(), max="Present + 48 hours"
            ),
        ),
    ),
    Coordinate(
        name="ingested_forecast_length",
        encoding=Encoding(
            dtype="int64",
            fill_value=-1,
            # TODO
            # compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
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
            # TODO
            # compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
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
