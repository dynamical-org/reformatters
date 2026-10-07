import re

import pandas as pd

from reformatters.common.types import Timedelta, Timestamp
from reformatters.google.weathernext_virtual.listing import PROXY_LOCATION_PREFIX

from .template_config import STATISTICS, VARIABLE_SPECS

SOURCE_PREFIX = "gs://weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr/2026_to_present/"
_SOURCE_ARRAYS = frozenset(
    f"{base}_{statistic}"
    for base in (
        *(spec[1] for spec in VARIABLE_SPECS),
        "station_head_temperature_2m",
        "station_head_dewpoint_temperature_2m",
    )
    for statistic in STATISTICS
)


def source_horizon(init_time: Timestamp) -> Timedelta:
    return pd.Timedelta(hours=360 if init_time.hour % 6 == 0 else 48)


def source_store_url(init_time: Timestamp) -> str:
    return f"{SOURCE_PREFIX}{init_time:%Y%m%d_%H}hr_01_preds/predictions.zarr"


def source_store_key(init_time: Timestamp) -> str:
    return source_store_url(init_time).split("/", 3)[3]


def parse_wn3_source_location(location: str) -> tuple[Timestamp, Timedelta]:
    prefix = PROXY_LOCATION_PREFIX + SOURCE_PREFIX.split("/", 3)[3]
    assert location.startswith(prefix), "Unexpected WeatherNext 3 proxy prefix"
    match = re.fullmatch(
        r"([0-9]{8})_([0-9]{2})hr_01_preds/predictions\.zarr/([a-z0-9_]+)/c/(0|[1-9][0-9]*)/0/0",
        location.removeprefix(prefix),
    )
    assert match is not None, "Invalid WeatherNext 3 source key"
    date, hour, array, index = match.groups()
    assert array in _SOURCE_ARRAYS, "Unknown source array"
    init_time = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]}T{hour}:00")
    assert init_time >= pd.Timestamp("2026-01-01"), "Invalid source init"
    assert int(index) < source_horizon(init_time) // pd.Timedelta("1h"), (
        "Lead exceeds source horizon"
    )
    return init_time, pd.Timedelta(hours=int(index) + 1)
