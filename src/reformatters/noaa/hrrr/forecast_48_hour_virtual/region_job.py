from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.hrrr.forecast_virtual_region_job import (
    NoaaHrrrForecastVirtualRegionJob,
)


class NoaaHrrrForecast48HourVirtualRegionJob(NoaaHrrrForecastVirtualRegionJob):
    """RegionJob for the HRRR 48-hour virtual forecast dataset."""

    # 14h = two 6h cycles back + ~2h publication slack, so a couple of missed runs
    # still self-heal.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("14h")
