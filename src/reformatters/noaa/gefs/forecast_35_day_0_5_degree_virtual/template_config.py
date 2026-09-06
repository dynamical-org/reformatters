import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.types import Timedelta
from reformatters.noaa.gefs.gefs_config_models import GEFS_EXTENSION_MAX
from reformatters.noaa.gefs.virtual_template_config import (
    NoaaGefsForecastABVirtualTemplateConfig,
)


class NoaaGefsForecast35Day05DegreeVirtualTemplateConfig(
    NoaaGefsForecastABVirtualTemplateConfig
):
    """Virtual GEFS 35 day forecast: every 0.5 degree pgrb2a and pgrb2b message of all
    31 ensemble members, out to the 840 hour lead only the 00z cycle reaches.

    Only 00z runs that far, so the init axis is daily where the 16 day dataset's is 6
    hourly, and every init this dataset serves is also served there through 384 hours.
    """

    forecast_length: Timedelta = GEFS_EXTENSION_MAX
    append_dim_frequency: Timedelta = pd.Timedelta("24h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return self._dataset_attributes(
            dataset_id="noaa-gefs-forecast-35-day-0-5-degree-virtual",
            dataset_version="0.1.0",
            name="NOAA GEFS forecast 35 day 0.5 degree, virtual",
            description=(
                "Weather forecasts from the Global Ensemble Forecast System (GEFS) "
                "operated by NOAA NWS NCEP, served as references to the source GRIB "
                "messages. Covers every variable the 0.5 degree pgrb2a and pgrb2b "
                "files carry, for all 31 ensemble members, out to the 840 hour lead "
                "time the 00z cycle reaches."
            ),
        )
