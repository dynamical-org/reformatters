from typing import ClassVar

import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.common.types import Dims, Timedelta
from reformatters.noaa.rrfs.template_config import NoaaRrfsForecastTemplateConfig


class NoaaRrfsForecastSubHourlyVirtualTemplateConfig(NoaaRrfsForecastTemplateConfig):
    dataset_id: ClassVar[str] = "noaa-rrfs-forecast-sub-hourly-virtual"
    dataset_name: ClassVar[str] = "NOAA RRFS forecast, sub-hourly, virtual"
    forecast_length: Timedelta = pd.Timedelta("18h")
    append_dim_frequency: Timedelta = pd.Timedelta("1h")
    dims: Dims = {ROOT: ("init_time", "lead_time", "y", "x")}
    sub_hourly: bool = True
    lead_frequency: Timedelta = pd.Timedelta("15min")
    first_lead: Timedelta = pd.Timedelta("15min")
