from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsEnsForecastVirtualRegionJob
from .template_config import NoaaRrfsEnsForecastVirtualTemplateConfig


class NoaaRrfsEnsForecastVirtualDataset(NoaaRrfsDataset):
    # Poll before early leads while preserving late-cycle coverage.
    update_offset_minutes: NonNegativeInt = 70
    update_deadline: timedelta = timedelta(minutes=192)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=6)
    retry_late_sources: ClassVar[bool] = True
    template_config: NoaaRrfsEnsForecastVirtualTemplateConfig = (
        NoaaRrfsEnsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsEnsForecastVirtualRegionJob] = (
        NoaaRrfsEnsForecastVirtualRegionJob
    )
