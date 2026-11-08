from reformatters.ecmwf.ifs_ens.virtual_dynamical_dataset import (
    EcmwfIfsEnsForecastVirtualDataset,
)

from .region_job import EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob
from .template_config import EcmwfIfsEnsForecast6Day01DegreeVirtualTemplateConfig


class EcmwfIfsEnsForecast6Day01DegreeVirtualDataset(EcmwfIfsEnsForecastVirtualDataset):
    template_config: EcmwfIfsEnsForecast6Day01DegreeVirtualTemplateConfig = (
        EcmwfIfsEnsForecast6Day01DegreeVirtualTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob] = (
        EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob
    )
