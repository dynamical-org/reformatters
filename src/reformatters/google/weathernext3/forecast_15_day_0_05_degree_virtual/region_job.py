from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.google.weathernext3.forecast_virtual.region_job import (
    GoogleWeathernext3ForecastVirtualRegionJob,
)


class GoogleWeathernext3Forecast15Day005DegreeVirtualRegionJob(
    GoogleWeathernext3ForecastVirtualRegionJob
):
    init_frequency: ClassVar[Timedelta] = pd.Timedelta("6h")
    manifest_init_split: ClassVar[int] = 64
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta(
        "360h"
    ) + pd.Timedelta("2D")
