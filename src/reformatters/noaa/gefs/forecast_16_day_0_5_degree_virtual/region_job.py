from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.gefs.virtual_region_job import NoaaGefsForecastVirtualRegionJob


class NoaaGefsForecast16Day05DegreeVirtualRegionJob(NoaaGefsForecastVirtualRegionJob):
    """RegionJob for the GEFS 16 day 0.5 degree virtual forecast dataset."""

    # Three update cron fires' span, so two consecutive missed runs still self-heal.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("18h")
