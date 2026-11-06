from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig, FastModel


class EcmwfIfsEnsForecast025DegreeFastTemplateConfig(EcmwfFastTemplateConfig):
    model: FastModel = "ifs"

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-ifs-ens-forecast-0-25-degree-fast",
            dataset_version="0.1.0",
            name="ECMWF IFS ENS forecast, 0.25 degree, fast",
            description="Weather forecasts from the ECMWF IFS ENS, published with low latency.",
            attribution="ECMWF IFS ENS forecast data processed by dynamical.org.",
            license="LicenseRef-Commercial",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain="Most recent 12 forecast initializations",
            time_resolution="Forecasts initialized every 6 hours",
            forecast_domain="Forecast lead time 0-360 hours (0-15 days) ahead",
            forecast_resolution="0-90 hours: hourly, 93-144 hours: 3 hourly, 150-360 hours: 6 hourly",
        )
