from reformatters.ecmwf.fast.dynamical_dataset import EcmwfFastDataset

from .region_job import EcmwfAifsEnsForecastFastRegionJob
from .template_config import EcmwfAifsEnsForecastFastTemplateConfig


class EcmwfAifsEnsForecastFastDataset(EcmwfFastDataset):
    template_config: EcmwfAifsEnsForecastFastTemplateConfig = (
        EcmwfAifsEnsForecastFastTemplateConfig()
    )
    region_job_class: type[EcmwfAifsEnsForecastFastRegionJob] = (
        EcmwfAifsEnsForecastFastRegionJob
    )
