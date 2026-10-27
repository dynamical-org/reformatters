from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.common.kubernetes import ConcurrencyPolicy
from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecastSubHourlyVirtualRegionJob
from .template_config import NoaaRrfsForecastSubHourlyVirtualTemplateConfig


class NoaaRrfsForecastSubHourlyVirtualDataset(NoaaRrfsDataset):
    update_offset_minutes: NonNegativeInt = 75
    update_deadline: timedelta = timedelta(minutes=62, seconds=37)
    concurrency_policy: ClassVar[ConcurrencyPolicy] = "Allow"
    max_delay: ClassVar[timedelta] = timedelta(minutes=138)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(seconds=81)
    # 00Z and 12Z source publication can continue past the poll deadline.
    min_present_fraction: ClassVar[tuple[float, ...]] = (6 / 18, 1.0)
    template_config: NoaaRrfsForecastSubHourlyVirtualTemplateConfig = (
        NoaaRrfsForecastSubHourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecastSubHourlyVirtualRegionJob] = (
        NoaaRrfsForecastSubHourlyVirtualRegionJob
    )
