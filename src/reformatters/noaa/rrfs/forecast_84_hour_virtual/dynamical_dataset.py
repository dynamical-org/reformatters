from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.common.kubernetes import ConcurrencyPolicy
from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast84HourVirtualRegionJob
from .template_config import NoaaRrfsForecast84HourVirtualTemplateConfig


class NoaaRrfsForecast84HourVirtualDataset(NoaaRrfsDataset):
    update_offset_minutes: NonNegativeInt = 71
    update_deadline: timedelta = timedelta(hours=6, minutes=5, seconds=10)
    concurrency_policy: ClassVar[ConcurrencyPolicy] = "Allow"
    max_delay: ClassVar[timedelta] = timedelta(minutes=133)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=3, seconds=23)
    # Snow density is all NaN when the air 300 m above ground is 5 °C or warmer everywhere.
    allow_all_nan_vars: ClassVar[tuple[str, ...]] = ("new_snow_density_surface",)
    template_config: NoaaRrfsForecast84HourVirtualTemplateConfig = (
        NoaaRrfsForecast84HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast84HourVirtualRegionJob] = (
        NoaaRrfsForecast84HourVirtualRegionJob
    )
