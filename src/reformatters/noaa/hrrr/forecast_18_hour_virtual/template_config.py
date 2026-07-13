import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.types import Timedelta
from reformatters.noaa.hrrr.forecast_virtual_template_config import (
    NoaaHrrrForecastVirtualTemplateConfig,
)


class NoaaHrrrForecast18HourVirtualTemplateConfig(
    NoaaHrrrForecastVirtualTemplateConfig
):
    """Virtual HRRR 18-hour forecast: every hourly cycle, which all reach f18."""

    forecast_length: Timedelta = pd.Timedelta("18h")
    append_dim_frequency: Timedelta = pd.Timedelta("1h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return self._dataset_attributes(
            dataset_id="noaa-hrrr-forecast-18-hour-virtual",
            dataset_version="0.1.0",
            name="NOAA HRRR forecast, 18 hour, virtual",
        )
