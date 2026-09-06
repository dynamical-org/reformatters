from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.gefs.virtual_region_job import NoaaGefsForecastVirtualRegionJob


class NoaaGefsForecast35Day05DegreeVirtualRegionJob(NoaaGefsForecastVirtualRegionJob):
    """RegionJob for the GEFS 35 day 0.5 degree virtual forecast dataset."""

    # Three update cron fires' span, so two consecutive missed runs still self-heal. A
    # cycle's longest leads publish until init+28h, past the next fire, so the window
    # has to reach back more than one init even with none missed.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("72h")
