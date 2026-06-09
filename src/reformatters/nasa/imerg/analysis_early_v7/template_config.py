from pydantic import computed_field

from reformatters.common.config_models import DatasetAttributes
from reformatters.nasa.imerg.template_config import NasaImergAnalysisTemplateConfig


class NasaImergAnalysisEarlyV7TemplateConfig(NasaImergAnalysisTemplateConfig):
    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="nasa-imerg-analysis-early-v7",
            dataset_version="0.1.0",
            name="NASA IMERG Early Run analysis, half-hourly, V07",
            description="Global half-hourly precipitation estimates from the NASA GPM IMERG Early Run (V07), a low-latency multi-satellite precipitation product.",
            attribution="NASA GPM IMERG data processed by dynamical.org from NASA GES DISC.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.1 degrees (~10km)",
            time_domain=f"{self.append_dim_start} UTC to Present",
            time_resolution="30 minutes",
        )
