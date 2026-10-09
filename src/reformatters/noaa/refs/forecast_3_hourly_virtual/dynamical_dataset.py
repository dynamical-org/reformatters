from reformatters.noaa.refs.dynamical_dataset import NoaaRefsDataset

from .region_job import NoaaRefsForecast3HourlyVirtualRegionJob
from .template_config import NoaaRefsForecast3HourlyVirtualTemplateConfig


class NoaaRefsForecast3HourlyVirtualDataset(NoaaRefsDataset):
    template_config: NoaaRefsForecast3HourlyVirtualTemplateConfig = (
        NoaaRefsForecast3HourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsForecast3HourlyVirtualRegionJob] = (
        NoaaRefsForecast3HourlyVirtualRegionJob
    )
