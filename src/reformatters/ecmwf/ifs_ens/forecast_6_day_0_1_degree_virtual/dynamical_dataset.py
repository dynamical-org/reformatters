from datetime import timedelta
from typing import ClassVar

from reformatters.ecmwf.ifs_ens.virtual_dynamical_dataset import (
    EcmwfIfsEnsForecastVirtualDataset,
)

from .region_job import EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob
from .template_config import EcmwfIfsEnsForecast6Day01DegreeVirtualTemplateConfig


class EcmwfIfsEnsForecast6Day01DegreeVirtualDataset(EcmwfIfsEnsForecastVirtualDataset):
    schedule: ClassVar[str] = "12 2,8,14,20 * * *"
    deadline: ClassVar[timedelta] = timedelta(hours=6, minutes=6)
    max_delay: ClassVar[timedelta] = timedelta(hours=9, minutes=42)
    template_config: EcmwfIfsEnsForecast6Day01DegreeVirtualTemplateConfig = (
        EcmwfIfsEnsForecast6Day01DegreeVirtualTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob] = (
        EcmwfIfsEnsForecast6Day01DegreeVirtualRegionJob
    )
