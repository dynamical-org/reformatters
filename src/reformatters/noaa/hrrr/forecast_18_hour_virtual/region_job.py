from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualRegionJob,
)


class NoaaHrrrForecast18HourVirtualRegionJob(NoaaHrrrForecastVirtualRegionJob):
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("6h")
