from typing import ClassVar, Literal

import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.pydantic import replace
from reformatters.common.types import Timestamp
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)
from reformatters.google.weathernext_terms import (
    COPYRIGHT,
    LICENSE,
    REAL_TIME_ATTRIBUTION,
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
        "model."
    )

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return replace(
            super().dataset_attributes,
            license=LICENSE,
            attribution=(
                "Google WeatherNext 2 forecast data processed by dynamical.org "
                "from Google's WeatherNext 2 experimental data. "
                f"{COPYRIGHT}. {REAL_TIME_ATTRIBUTION}"
            ),
        )
