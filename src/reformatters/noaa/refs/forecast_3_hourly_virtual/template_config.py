from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.refs.template_config import NoaaRefsForecastTemplateConfig


class NoaaRefsForecast3HourlyVirtualTemplateConfig(NoaaRefsForecastTemplateConfig):
    dataset_id: ClassVar[str] = "noaa-refs-forecast-3-hourly-virtual"
    dataset_name: ClassVar[str] = "NOAA REFS forecast, 3 hourly, virtual"
    first_lead: Timedelta = pd.Timedelta("3h")
    lead_frequency: Timedelta = pd.Timedelta("3h")
