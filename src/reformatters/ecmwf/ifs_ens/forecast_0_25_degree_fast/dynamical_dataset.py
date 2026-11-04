from reformatters.ecmwf.fast.dynamical_dataset import EcmwfFastDataset

from .region_job import EcmwfIfsEnsForecast025DegreeFastRegionJob
from .template_config import EcmwfIfsEnsForecast025DegreeFastTemplateConfig


class EcmwfIfsEnsForecast025DegreeFastDataset(EcmwfFastDataset):
    template_config: EcmwfIfsEnsForecast025DegreeFastTemplateConfig = (
        EcmwfIfsEnsForecast025DegreeFastTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast025DegreeFastRegionJob] = (
        EcmwfIfsEnsForecast025DegreeFastRegionJob
    )
