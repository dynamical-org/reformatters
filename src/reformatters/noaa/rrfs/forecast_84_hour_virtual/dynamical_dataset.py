from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast84HourVirtualRegionJob
from .template_config import NoaaRrfsForecast84HourVirtualTemplateConfig


class NoaaRrfsForecast84HourVirtualDataset(NoaaRrfsDataset):
    # Files publish near init +108-212 min; polling starts before the first lead.
    update_offset_minutes: NonNegativeInt = 100
    update_deadline: timedelta = timedelta(minutes=135)
    # Snow density is all NaN when the air 300 m above ground is 5 °C or warmer everywhere.
    allow_all_nan_vars: ClassVar[tuple[str, ...]] = ("new_snow_density_surface",)
    template_config: NoaaRrfsForecast84HourVirtualTemplateConfig = (
        NoaaRrfsForecast84HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast84HourVirtualRegionJob] = (
        NoaaRrfsForecast84HourVirtualRegionJob
    )
