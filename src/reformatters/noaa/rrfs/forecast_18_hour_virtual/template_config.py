from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta, Timestamp
from reformatters.noaa.rrfs.template_config import (
    NoaaRrfsForecastTemplateConfig,
)


class NoaaRrfsForecast18HourVirtualTemplateConfig(NoaaRrfsForecastTemplateConfig):
    dataset_id: ClassVar[str] = "noaa-rrfs-forecast-18-hour-virtual"
    dataset_name: ClassVar[str] = "NOAA RRFS forecast, 18 hour, virtual"
    forecast_length: Timedelta = pd.Timedelta("18h")
    append_dim_frequency: Timedelta = pd.Timedelta("3h")
    append_dim_start: Timestamp = pd.Timestamp("2026-08-13T03:00")
