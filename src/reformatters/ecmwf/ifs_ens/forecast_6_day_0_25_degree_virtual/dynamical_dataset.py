from datetime import timedelta
from typing import ClassVar

from reformatters.ecmwf.ifs_ens.virtual_dynamical_dataset import (
    EcmwfIfsEnsForecastVirtualDataset,
)

from .region_job import EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob
from .template_config import EcmwfIfsEnsForecast6Day025DegreeVirtualTemplateConfig


class EcmwfIfsEnsForecast6Day025DegreeVirtualDataset(EcmwfIfsEnsForecastVirtualDataset):
    schedule: ClassVar[str] = "12 0,6,12,18 * * *"
    deadline: ClassVar[timedelta] = timedelta(hours=6, minutes=6)
    max_delay: ClassVar[timedelta] = timedelta(hours=7, minutes=42)
    template_config: EcmwfIfsEnsForecast6Day025DegreeVirtualTemplateConfig = (
        EcmwfIfsEnsForecast6Day025DegreeVirtualTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob] = (
        EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob
    )
