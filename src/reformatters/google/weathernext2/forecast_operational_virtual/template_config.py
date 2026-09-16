from typing import ClassVar, Literal

import pandas as pd

from reformatters.common.types import Timestamp
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)


class GoogleWeathernext2ForecastOperationalVirtualTemplateConfig(
    GoogleWeathernext2ForecastVirtualTemplateConfig
):
    source_layout: Literal["operational"] = "operational"
    append_dim_start: Timestamp = pd.Timestamp("2025-01-01T00:00")
    dataset_id_value: ClassVar[str] = "google-weathernext2-forecast-operational-virtual"
    dataset_name_value: ClassVar[str] = (
        "Google WeatherNext 2 operational forecast, virtual"
    )
    time_domain_end: ClassVar[str] = "Present"
    dataset_description: ClassVar[str] = (
        "Weather forecasts from the 64-member Google DeepMind WeatherNext 2 ensemble "
        "model. Forecast values are published once their valid time is at least one "
        "hour in the past, so recent initialization times are intentionally partial."
    )
