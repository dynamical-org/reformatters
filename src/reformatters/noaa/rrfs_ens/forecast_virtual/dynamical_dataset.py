from datetime import timedelta

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsEnsForecastVirtualRegionJob
from .template_config import NoaaRrfsEnsForecastVirtualTemplateConfig


class NoaaRrfsEnsForecastVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +81-212 min; polling starts before the first lead.
    update_offset_minutes: NonNegativeInt = 75
    update_deadline: timedelta = timedelta(minutes=160)
    template_config: NoaaRrfsEnsForecastVirtualTemplateConfig = (
        NoaaRrfsEnsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsEnsForecastVirtualRegionJob] = (
        NoaaRrfsEnsForecastVirtualRegionJob
    )
