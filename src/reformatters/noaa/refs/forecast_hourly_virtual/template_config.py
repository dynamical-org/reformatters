from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.refs.template_config import NoaaRefsForecastTemplateConfig


class NoaaRefsForecastHourlyVirtualTemplateConfig(NoaaRefsForecastTemplateConfig):
    dataset_id: ClassVar[str] = "noaa-refs-forecast-hourly-virtual"
    dataset_name: ClassVar[str] = "NOAA REFS forecast, hourly, virtual"
    first_lead: Timedelta = pd.Timedelta("1h")
    lead_frequency: Timedelta = pd.Timedelta("1h")
