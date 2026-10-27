from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.common.kubernetes import ConcurrencyPolicy
from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecast18HourVirtualRegionJob
from .template_config import NoaaRrfsForecast18HourVirtualTemplateConfig


class NoaaRrfsForecast18HourVirtualDataset(NoaaRrfsDataset):
    update_offset_minutes: NonNegativeInt = 70
    update_deadline: timedelta = timedelta(hours=3, minutes=4, seconds=24)
    concurrency_policy: ClassVar[ConcurrencyPolicy] = "Allow"
    max_delay: ClassVar[timedelta] = timedelta(minutes=133)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=3, seconds=8)
    # Snow density is all NaN when the air 300 m above ground is 5 °C or warmer everywhere.
    allow_all_nan_vars: ClassVar[tuple[str, ...]] = ("new_snow_density_surface",)
    template_config: NoaaRrfsForecast18HourVirtualTemplateConfig = (
        NoaaRrfsForecast18HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecast18HourVirtualRegionJob] = (
        NoaaRrfsForecast18HourVirtualRegionJob
    )
