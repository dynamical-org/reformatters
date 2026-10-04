from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta, Timestamp
from reformatters.noaa.rrfs.template_config import (
    NoaaRrfsForecastTemplateConfig,
)


class NoaaRrfsForecast84HourVirtualTemplateConfig(NoaaRrfsForecastTemplateConfig):
    dataset_id: ClassVar[str] = "noaa-rrfs-forecast-84-hour-virtual"
    dataset_name: ClassVar[str] = "NOAA RRFS forecast, 84 hour, virtual"
    forecast_length: Timedelta = pd.Timedelta("84h")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")
    append_dim_start: Timestamp = pd.Timestamp("2026-08-13T06:00")
