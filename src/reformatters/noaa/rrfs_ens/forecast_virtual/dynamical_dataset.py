from pydantic import NonNegativeInt, PositiveInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsEnsForecastVirtualRegionJob
from .template_config import NoaaRrfsEnsForecastVirtualTemplateConfig


class NoaaRrfsEnsForecastVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +81-212 min; polling starts before the first lead.
    update_offset_minutes: NonNegativeInt = 75
    update_deadline_minutes: PositiveInt = 160
    template_config: NoaaRrfsEnsForecastVirtualTemplateConfig = (
        NoaaRrfsEnsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsEnsForecastVirtualRegionJob] = (
        NoaaRrfsEnsForecastVirtualRegionJob
    )
