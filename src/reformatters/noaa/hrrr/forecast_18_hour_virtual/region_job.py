from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.hrrr.forecast_virtual_region_job import (
    NoaaHrrrForecastVirtualRegionJob,
)


class NoaaHrrrForecast18HourVirtualRegionJob(NoaaHrrrForecastVirtualRegionJob):
    """RegionJob for the HRRR 18-hour virtual forecast dataset."""

    # 6h = a few 1h cycles back + ~2h f18 publication slack, so several missed or
    # deadline-killed hourly runs still self-heal.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("6h")
