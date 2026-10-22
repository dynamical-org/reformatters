from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .models import NoaaRefsDataVar
from .region_job import NoaaRefsRegionJob, NoaaRefsSourceFileCoord
from .template_config import NoaaRefsForecastTemplateConfig


class NoaaRefsDataset(NoaaRrfsDataset[NoaaRefsDataVar, NoaaRefsSourceFileCoord]):
    update_offset_minutes: NonNegativeInt = 105
    update_deadline: timedelta = timedelta(minutes=163)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=6)
    retry_late_sources: ClassVar[bool] = True
    template_config: NoaaRefsForecastTemplateConfig
    region_job_class: type[NoaaRefsRegionJob] = NoaaRefsRegionJob
