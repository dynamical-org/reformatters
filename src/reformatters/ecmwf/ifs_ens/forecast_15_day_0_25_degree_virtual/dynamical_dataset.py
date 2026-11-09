from datetime import timedelta
from typing import ClassVar

from reformatters.ecmwf.ifs_ens.virtual_dynamical_dataset import (
    EcmwfIfsEnsForecastVirtualDataset,
)

from .region_job import EcmwfIfsEnsForecast15Day025DegreeVirtualRegionJob
from .template_config import EcmwfIfsEnsForecast15Day025DegreeVirtualTemplateConfig


class EcmwfIfsEnsForecast15Day025DegreeVirtualDataset(
    EcmwfIfsEnsForecastVirtualDataset
):
    schedule: ClassVar[str] = "19 7,19 * * *"
    deadline: ClassVar[timedelta] = timedelta(hours=12, minutes=6)
    max_delay: ClassVar[timedelta] = timedelta(hours=7, minutes=42)
    template_config: EcmwfIfsEnsForecast15Day025DegreeVirtualTemplateConfig = (
        EcmwfIfsEnsForecast15Day025DegreeVirtualTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast15Day025DegreeVirtualRegionJob] = (
        EcmwfIfsEnsForecast15Day025DegreeVirtualRegionJob
    )
