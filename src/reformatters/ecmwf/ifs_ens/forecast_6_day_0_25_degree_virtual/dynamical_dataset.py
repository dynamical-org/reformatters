from reformatters.ecmwf.ifs_ens.virtual_dynamical_dataset import (
    EcmwfIfsEnsForecastVirtualDataset,
)

from .region_job import EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob
from .template_config import EcmwfIfsEnsForecast6Day025DegreeVirtualTemplateConfig


class EcmwfIfsEnsForecast6Day025DegreeVirtualDataset(EcmwfIfsEnsForecastVirtualDataset):
    template_config: EcmwfIfsEnsForecast6Day025DegreeVirtualTemplateConfig = (
        EcmwfIfsEnsForecast6Day025DegreeVirtualTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob] = (
        EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob
    )
