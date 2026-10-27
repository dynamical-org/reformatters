from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.common.kubernetes import ConcurrencyPolicy
from reformatters.noaa.refs.dynamical_dataset import NoaaRefsDataset

from .region_job import NoaaRefsForecast3HourlyVirtualRegionJob
from .template_config import NoaaRefsForecast3HourlyVirtualTemplateConfig


class NoaaRefsForecast3HourlyVirtualDataset(NoaaRefsDataset):
    update_offset_minutes: NonNegativeInt = 110
    update_deadline: timedelta = timedelta(hours=6, minutes=2, seconds=55)
    concurrency_policy: ClassVar[ConcurrencyPolicy] = "Allow"
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=1, seconds=39)
    template_config: NoaaRefsForecast3HourlyVirtualTemplateConfig = (
        NoaaRefsForecast3HourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsForecast3HourlyVirtualRegionJob] = (
        NoaaRefsForecast3HourlyVirtualRegionJob
    )

    def current_data_max_delay(self) -> timedelta:
        return timedelta(minutes=156)
