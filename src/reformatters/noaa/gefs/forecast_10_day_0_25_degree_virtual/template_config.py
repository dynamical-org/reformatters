from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.common.types import Timedelta
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_S_FILE_MAX,
    GEFSSourceFileType,
)
from reformatters.noaa.gefs.virtual_template_config import (
    NoaaGefsForecastVirtualTemplateConfig,
)


class NoaaGefsForecast10Day025DegreeVirtualTemplateConfig(
    NoaaGefsForecastVirtualTemplateConfig
):
    """Virtual GEFS 10 day forecast: every 0.25 degree pgrb2s message of all 31
    ensemble members, out to the 240 hour lead where that file ends."""

    source_file_types: frozenset[GEFSSourceFileType] = frozenset({"s"})
    forecast_length: Timedelta = GEFS_S_FILE_MAX

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return self._dataset_attributes(
            dataset_id="noaa-gefs-forecast-10-day-0-25-degree-virtual",
            dataset_version="0.1.0",
            name="NOAA GEFS forecast, 10 day, 0.25 degree, virtual",
            description=(
                "Weather forecasts from the Global Ensemble Forecast System (GEFS) "
                "operated by NOAA NWS NCEP, served as references to the source GRIB "
                "messages. Covers every variable the 0.25 degree pgrb2s file carries, "
                "for all 31 ensemble members, through the 240 hour lead time where "
                "that file ends; the materialized noaa-gefs-forecast-35-day continues "
                "past it on a coarser grid."
            ),
        )
