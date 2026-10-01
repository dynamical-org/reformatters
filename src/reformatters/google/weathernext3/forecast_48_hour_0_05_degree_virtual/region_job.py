from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.google.weathernext3.forecast_virtual.region_job import (
    GoogleWeathernext3ForecastVirtualRegionJob,
)


class GoogleWeathernext3Forecast48Hour005DegreeVirtualRegionJob(
    GoogleWeathernext3ForecastVirtualRegionJob
):
    init_frequency: ClassVar[Timedelta] = pd.Timedelta("1h")
    manifest_init_split: ClassVar[int] = 384
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("48h") + pd.Timedelta(
        "2D"
    )
