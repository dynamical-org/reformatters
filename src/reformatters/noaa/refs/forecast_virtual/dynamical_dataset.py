from typing import ClassVar

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRefsForecastVirtualRegionJob
from .template_config import NoaaRefsForecastVirtualTemplateConfig


class NoaaRefsForecastVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +81-212 min; polling starts before the first lead.
    _operational_timing: ClassVar[tuple[int, int]] = (75, 160)
    template_config: NoaaRefsForecastVirtualTemplateConfig = (
        NoaaRefsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsForecastVirtualRegionJob] = (
        NoaaRefsForecastVirtualRegionJob
    )
