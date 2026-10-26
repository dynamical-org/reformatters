import pandas as pd

from reformatters.common.types import Timedelta, Timestamp

SOURCE_PREFIX = "gs://weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr/2026_to_present/"


def source_horizon(init_time: Timestamp) -> Timedelta:
    return pd.Timedelta(hours=360 if init_time.hour % 6 == 0 else 48)


def source_store_url(init_time: Timestamp) -> str:
    return f"{SOURCE_PREFIX}{init_time:%Y%m%d_%H}hr_01_preds/predictions.zarr"


def source_store_key(init_time: Timestamp) -> str:
    return source_store_url(init_time).split("/", 3)[3]
