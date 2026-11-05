from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig, FastModel


class EcmwfAifsEnsForecastFastTemplateConfig(EcmwfFastTemplateConfig):
    model: FastModel = "aifs"

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-aifs-ens-forecast-fast",
            dataset_version="0.1.0",
            name="ECMWF AIFS ENS forecast, fast",
            description="Weather forecasts from the ECMWF AIFS ensemble model, published with low latency.",
            attribution="ECMWF AIFS ensemble forecast data processed by dynamical.org.",
            license="LicenseRef-Commercial",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution="Forecasts initialized every 6 hours",
            forecast_domain="Forecast lead time 0-360 hours (0-15 days) ahead",
            forecast_resolution="6 hourly",
        )
