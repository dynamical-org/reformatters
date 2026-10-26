from datetime import timedelta
from typing import ClassVar

from pydantic import NonNegativeInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .models import NoaaRefsDataVar
from .region_job import NoaaRefsRegionJob, NoaaRefsSourceFileCoord
from .template_config import NoaaRefsForecastTemplateConfig


class NoaaRefsDataset(NoaaRrfsDataset[NoaaRefsDataVar, NoaaRefsSourceFileCoord]):
    update_offset_minutes: NonNegativeInt = 105
    # Poll until init + 6 h to cover late cycles, ending before the next fire.
    update_deadline: timedelta = timedelta(minutes=262)
    # Validation of the hourly product takes up to about 4 minutes.
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=7)
    template_config: NoaaRefsForecastTemplateConfig
    region_job_class: type[NoaaRefsRegionJob] = NoaaRefsRegionJob
