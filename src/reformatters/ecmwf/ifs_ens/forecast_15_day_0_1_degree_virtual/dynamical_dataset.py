from datetime import timedelta
from typing import ClassVar

from reformatters.ecmwf.ifs_ens.virtual_dynamical_dataset import (
    EcmwfIfsEnsForecastVirtualDataset,
)

from .region_job import EcmwfIfsEnsForecast15Day01DegreeVirtualRegionJob
from .template_config import EcmwfIfsEnsForecast15Day01DegreeVirtualTemplateConfig


class EcmwfIfsEnsForecast15Day01DegreeVirtualDataset(EcmwfIfsEnsForecastVirtualDataset):
    schedule: ClassVar[str] = "19 9,21 * * *"
    deadline: ClassVar[timedelta] = timedelta(hours=12, minutes=6)
    max_delay: ClassVar[timedelta] = timedelta(hours=9, minutes=42)
    template_config: EcmwfIfsEnsForecast15Day01DegreeVirtualTemplateConfig = (
        EcmwfIfsEnsForecast15Day01DegreeVirtualTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast15Day01DegreeVirtualRegionJob] = (
        EcmwfIfsEnsForecast15Day01DegreeVirtualRegionJob
    )
