from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecastSubHourlyVirtualRegionJob
from .template_config import NoaaRrfsForecastSubHourlyVirtualTemplateConfig


class NoaaRrfsForecastSubHourlyVirtualDataset(NoaaRrfsDataset):
    template_config: NoaaRrfsForecastSubHourlyVirtualTemplateConfig = (
        NoaaRrfsForecastSubHourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecastSubHourlyVirtualRegionJob] = (
        NoaaRrfsForecastSubHourlyVirtualRegionJob
    )
