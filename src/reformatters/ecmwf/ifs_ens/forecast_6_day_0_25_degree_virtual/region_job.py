from typing import ClassVar, Literal

import pandas as pd

from reformatters.common.types import Timedelta, Timestamp
from reformatters.ecmwf.ifs_ens.virtual_region_job import (
    EcmwfIfsEnsForecastVirtualRegionJob,
)


class EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob(
    EcmwfIfsEnsForecastVirtualRegionJob
):
    resolution: ClassVar[Literal["0p25", "0p1"]] = "0p25"
    s3_fallback_before: ClassVar[Timestamp | None] = pd.Timestamp("2025-02-13T00")
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("30h")
