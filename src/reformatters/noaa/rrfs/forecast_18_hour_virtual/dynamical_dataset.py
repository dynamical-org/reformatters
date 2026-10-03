from typing import ClassVar

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast18HourVirtualRegionJob
from .template_config import NoaaRrfsForecast18HourVirtualTemplateConfig


class NoaaRrfsForecast18HourVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +110-140 min; polling starts before the first lead.
    _operational_timing: ClassVar[tuple[int, int]] = (100, 60)
    template_config: NoaaRrfsForecast18HourVirtualTemplateConfig = (
        NoaaRrfsForecast18HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast18HourVirtualRegionJob] = (
        NoaaRrfsForecast18HourVirtualRegionJob
    )
