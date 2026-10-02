import pandas as pd

from reformatters.common.types import Timedelta, Timestamp

PUBLICATION_HOLDBACK = pd.Timedelta("1h")


def utc_now() -> Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def is_publishable(
    init_time: Timestamp, lead_time: Timedelta, cutoff: Timestamp
) -> bool:
    return init_time + lead_time <= cutoff
