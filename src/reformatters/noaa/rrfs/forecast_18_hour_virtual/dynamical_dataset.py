from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast18HourVirtualRegionJob
from .template_config import NoaaRrfsForecast18HourVirtualTemplateConfig


class NoaaRrfsForecast18HourVirtualDataset(NoaaRrfsDataset):
    template_config: NoaaRrfsForecast18HourVirtualTemplateConfig = (
        NoaaRrfsForecast18HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast18HourVirtualRegionJob] = (
        NoaaRrfsForecast18HourVirtualRegionJob
    )
