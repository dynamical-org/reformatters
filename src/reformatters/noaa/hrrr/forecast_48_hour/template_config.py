from typing import Any, Literal

import numpy as np
import pandas as pd

from reformatters.common.config_models import DatasetAttributes
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

type Dim =                     Literal["init_time", "lead_time", "latitude", "longitude"]  # fmt: off
DIMS: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")  # fmt: off


# TODO: figure out latitude/longitude dimensions
def get_template_dimension_coordinates() -> dict[str, Any]:
    return {
        "init_time": get_init_time_coordinates(INIT_TIME_START + INIT_TIME_FREQUENCY),
        "lead_time": pd.timedelta_range("0h", "48h", freq=LEAD_TIME_FREQUENCY),
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
# These chunks are about XXXmb of uncompressed float32s
CHUNKS: dict[Dim, int] = {}

# SHARDS
# About XXXMB compressed, about XXGB uncompressed
SHARDS: dict[Dim, int] = {}

STATISTIC_VAR_CHUNKS_ORDERED = tuple(CHUNKS[dim] for dim in DIMS)
STATISTIC_VAR_SHARDS_ORDERED = tuple(SHARDS[dim] for dim in DIMS)
