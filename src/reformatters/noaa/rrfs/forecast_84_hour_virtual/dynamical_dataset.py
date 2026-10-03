from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast84HourVirtualRegionJob
from .template_config import NoaaRrfsForecast84HourVirtualTemplateConfig


class NoaaRrfsForecast84HourVirtualDataset(NoaaRrfsDataset):
    template_config: NoaaRrfsForecast84HourVirtualTemplateConfig = (
        NoaaRrfsForecast84HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast84HourVirtualRegionJob] = (
        NoaaRrfsForecast84HourVirtualRegionJob
    )
