from typing import ClassVar

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecastSubHourlyVirtualRegionJob
from .template_config import NoaaRrfsForecastSubHourlyVirtualTemplateConfig


class NoaaRrfsForecastSubHourlyVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +80-106 min; polling ends before the next cycle publishes.
    _operational_timing: ClassVar[tuple[int, int]] = (75, 55)
    template_config: NoaaRrfsForecastSubHourlyVirtualTemplateConfig = (
        NoaaRrfsForecastSubHourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecastSubHourlyVirtualRegionJob] = (
        NoaaRrfsForecastSubHourlyVirtualRegionJob
    )
