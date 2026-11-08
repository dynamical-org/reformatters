from reformatters.ecmwf.ifs_ens.virtual_dynamical_dataset import (
    EcmwfIfsEnsForecastVirtualDataset,
)

from .region_job import EcmwfIfsEnsForecast15Day01DegreeVirtualRegionJob
from .template_config import EcmwfIfsEnsForecast15Day01DegreeVirtualTemplateConfig


class EcmwfIfsEnsForecast15Day01DegreeVirtualDataset(EcmwfIfsEnsForecastVirtualDataset):
    template_config: EcmwfIfsEnsForecast15Day01DegreeVirtualTemplateConfig = (
        EcmwfIfsEnsForecast15Day01DegreeVirtualTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast15Day01DegreeVirtualRegionJob] = (
        EcmwfIfsEnsForecast15Day01DegreeVirtualRegionJob
    )
