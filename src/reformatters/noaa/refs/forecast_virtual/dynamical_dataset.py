from typing import ClassVar

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .models import NoaaRefsDataVar
from .region_job import NoaaRefsRegionJob, NoaaRefsSourceFileCoord
from .template_config import NoaaRefsForecastVirtualTemplateConfig


class NoaaRefsForecastVirtualDataset(
    NoaaRrfsDataset[NoaaRefsDataVar, NoaaRefsSourceFileCoord]
):
    _operational_timing: ClassVar[tuple[int, int]] = (115, 125)
    template_config: NoaaRefsForecastVirtualTemplateConfig = (
        NoaaRefsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsRegionJob] = NoaaRefsRegionJob
