from typing import ClassVar, Literal

import pandas as pd

from reformatters.common.types import DatetimeLike, Timestamp
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)


class GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig(
    GoogleWeathernext2ForecastVirtualTemplateConfig
):
    source_layout: Literal["historical"] = "historical"
    append_dim_start: Timestamp = pd.Timestamp("2022-01-01T00:00")
    dataset_id_value: ClassVar[str] = "google-weathernext2-forecast-historical-virtual"
    dataset_name_value: ClassVar[str] = (
        "Google WeatherNext 2 historical forecast, virtual"
    )
    time_domain_end: ClassVar[str] = "2024-12-31 18:00:00"
    init_time_statistics_max: ClassVar[str] = "2024-12-31T18:00:00"
    valid_time_statistics_max: ClassVar[str] = "2025-01-15T18:00:00"

    def append_dim_coordinates(self, end: DatetimeLike) -> pd.DatetimeIndex:
        historical_end = pd.Timestamp("2025-01-01T00:00")
        return super().append_dim_coordinates(min(pd.Timestamp(end), historical_end))
