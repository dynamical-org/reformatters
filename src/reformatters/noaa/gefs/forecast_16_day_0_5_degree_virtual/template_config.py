from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.types import Timedelta
from reformatters.noaa.gefs.gefs_config_models import GEFS_PRE_EXTENSION_MAX
from reformatters.noaa.gefs.virtual_template_config import (
    NoaaGefsForecastABVirtualTemplateConfig,
)


class NoaaGefsForecast16Day05DegreeVirtualTemplateConfig(
    NoaaGefsForecastABVirtualTemplateConfig
):
    """Virtual GEFS 16 day forecast: every 0.5 degree pgrb2a and pgrb2b message of all
    31 ensemble members, out to the 384 hour lead where those files end."""

    forecast_length: Timedelta = GEFS_PRE_EXTENSION_MAX

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return self._dataset_attributes(
            dataset_id="noaa-gefs-forecast-16-day-0-5-degree-virtual",
            dataset_version="0.1.0",
            name="NOAA GEFS forecast, 16 day, 0.5 degree, virtual",
            description=(
                "Weather forecasts from the Global Ensemble Forecast System (GEFS) "
                "operated by NOAA NWS NCEP."
            ),
        )
