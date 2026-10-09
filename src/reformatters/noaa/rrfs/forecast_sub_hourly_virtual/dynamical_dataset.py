from pydantic import NonNegativeInt, PositiveInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecastSubHourlyVirtualRegionJob
from .template_config import NoaaRrfsForecastSubHourlyVirtualTemplateConfig


class NoaaRrfsForecastSubHourlyVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +80-106 min; polling ends before the next cycle publishes.
    update_offset_minutes: NonNegativeInt = 75
    update_deadline_minutes: PositiveInt = 55
    template_config: NoaaRrfsForecastSubHourlyVirtualTemplateConfig = (
        NoaaRrfsForecastSubHourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecastSubHourlyVirtualRegionJob] = (
        NoaaRrfsForecastSubHourlyVirtualRegionJob
    )
