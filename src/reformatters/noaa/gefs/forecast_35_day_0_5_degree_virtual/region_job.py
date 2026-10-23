from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.gefs.virtual_region_job import NoaaGefsForecastVirtualRegionJob


class NoaaGefsForecast35Day05DegreeVirtualRegionJob(NoaaGefsForecastVirtualRegionJob):
    """RegionJob for the GEFS 35 day 0.5 degree virtual forecast dataset."""

    # Three daily inits retain the previous cycle's extension after a missed fire.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("72h")
