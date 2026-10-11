from datetime import timedelta

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast18HourVirtualRegionJob
from .template_config import NoaaRrfsForecast18HourVirtualTemplateConfig


class NoaaRrfsForecast18HourVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +110-140 min; polling starts before the first lead.
    update_offset_minutes: NonNegativeInt = 100
    update_deadline: timedelta = timedelta(minutes=60)
    template_config: NoaaRrfsForecast18HourVirtualTemplateConfig = (
        NoaaRrfsForecast18HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast18HourVirtualRegionJob] = (
        NoaaRrfsForecast18HourVirtualRegionJob
    )
