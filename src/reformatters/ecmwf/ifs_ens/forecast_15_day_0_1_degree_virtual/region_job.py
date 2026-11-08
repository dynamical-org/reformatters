from typing import ClassVar, Literal

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.ecmwf.ifs_ens.virtual_region_job import (
    EcmwfIfsEnsForecastVirtualRegionJob,
)


class EcmwfIfsEnsForecast15Day01DegreeVirtualRegionJob(
    EcmwfIfsEnsForecastVirtualRegionJob
):
    resolution: ClassVar[Literal["0p25", "0p1"]] = "0p1"
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("48h")
