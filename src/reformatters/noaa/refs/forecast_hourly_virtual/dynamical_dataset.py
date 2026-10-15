from reformatters.noaa.refs.dynamical_dataset import NoaaRefsDataset

from .region_job import NoaaRefsForecastHourlyVirtualRegionJob
from .template_config import NoaaRefsForecastHourlyVirtualTemplateConfig


class NoaaRefsForecastHourlyVirtualDataset(NoaaRefsDataset):
    template_config: NoaaRefsForecastHourlyVirtualTemplateConfig = (
        NoaaRefsForecastHourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsForecastHourlyVirtualRegionJob] = (
        NoaaRefsForecastHourlyVirtualRegionJob
    )
