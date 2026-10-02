from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.google.weathernext3.forecast_virtual.template_config import (
    GoogleWeathernext3ForecastVirtualTemplateConfig,
)


class GoogleWeathernext3Forecast15Day01DegreeVirtualTemplateConfig(
    GoogleWeathernext3ForecastVirtualTemplateConfig
):
    horizon_hours: ClassVar[int] = 360
    grid_degrees: ClassVar[float] = 0.1
    append_dim_frequency: Timedelta = pd.Timedelta("6h")
    dataset_id_value: ClassVar[str] = (
        "google-weathernext3-forecast-15-day-0-1-degree-virtual"
    )
    dataset_name_value: ClassVar[str] = (
        "Google WeatherNext 3 forecast, 15 day, 0.1 degree, virtual"
    )
