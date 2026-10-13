from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .region_job import NoaaRrfsForecastSubHourlyVirtualRegionJob
from .template_config import NoaaRrfsForecastSubHourlyVirtualTemplateConfig


class NoaaRrfsForecastSubHourlyVirtualDataset(NoaaRrfsDataset):
    # Polling leaves time to validate and exit before the next hourly update.
    update_offset_minutes: NonNegativeInt = 75
    update_deadline: timedelta = timedelta(minutes=59)
    update_job_deadline: timedelta | None = timedelta(minutes=59)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(seconds=60)
    # 00Z and 12Z cycles can have as few as 9/18 files by the poll deadline.
    min_present_fraction: ClassVar[tuple[float, ...]] = (6 / 18, 1.0)
    template_config: NoaaRrfsForecastSubHourlyVirtualTemplateConfig = (
        NoaaRrfsForecastSubHourlyVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRrfsForecastSubHourlyVirtualRegionJob] = (
        NoaaRrfsForecastSubHourlyVirtualRegionJob
    )
