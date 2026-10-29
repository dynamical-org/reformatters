from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.common.kubernetes import ConcurrencyPolicy
from reformatters.noaa.refs.dynamical_dataset import NoaaRefsDataset

from .region_job import NoaaRefsForecastHourlyVirtualRegionJob
from .template_config import NoaaRefsForecastHourlyVirtualTemplateConfig


class NoaaRefsForecastHourlyVirtualDataset(NoaaRefsDataset):
    update_offset_minutes: NonNegativeInt = 94
    update_deadline: timedelta = timedelta(hours=6, minutes=6)
    concurrency_policy: ClassVar[ConcurrencyPolicy] = "Allow"
    max_delay: ClassVar[timedelta] = timedelta(minutes=150)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=4, seconds=44)
    template_config: NoaaRefsForecastHourlyVirtualTemplateConfig = (
        NoaaRefsForecastHourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsForecastHourlyVirtualRegionJob] = (
        NoaaRefsForecastHourlyVirtualRegionJob
    )
