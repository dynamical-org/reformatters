from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.gefs.virtual_region_job import NoaaGefsForecastVirtualRegionJob


class NoaaGefsForecast10Day025DegreeVirtualRegionJob(NoaaGefsForecastVirtualRegionJob):
    """RegionJob for the GEFS 10 day 0.25 degree virtual forecast dataset."""

    # Three 6 hourly cycles, so two missed runs still self-heal.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("18h")
