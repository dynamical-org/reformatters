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
