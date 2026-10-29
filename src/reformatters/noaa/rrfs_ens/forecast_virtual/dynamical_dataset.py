from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.common.kubernetes import ConcurrencyPolicy
from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsEnsForecastVirtualRegionJob
from .template_config import NoaaRrfsEnsForecastVirtualTemplateConfig


class NoaaRrfsEnsForecastVirtualDataset(NoaaRrfsDataset):
    update_offset_minutes: NonNegativeInt = 56
    update_deadline: timedelta = timedelta(hours=6, minutes=5, seconds=16)
    concurrency_policy: ClassVar[ConcurrencyPolicy] = "Allow"
    max_delay: ClassVar[timedelta] = timedelta(minutes=122)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=4)
    template_config: NoaaRrfsEnsForecastVirtualTemplateConfig = (
        NoaaRrfsEnsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsEnsForecastVirtualRegionJob] = (
        NoaaRrfsEnsForecastVirtualRegionJob
    )
