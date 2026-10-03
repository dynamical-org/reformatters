from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRefsForecastVirtualRegionJob
from .template_config import NoaaRefsForecastVirtualTemplateConfig


class NoaaRefsForecastVirtualDataset(NoaaRrfsDataset):
    template_config: NoaaRefsForecastVirtualTemplateConfig = (
        NoaaRefsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsForecastVirtualRegionJob] = (
        NoaaRefsForecastVirtualRegionJob
    )
