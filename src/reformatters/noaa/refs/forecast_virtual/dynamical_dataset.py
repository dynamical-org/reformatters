from pydantic import NonNegativeInt, PositiveInt

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .models import NoaaRefsDataVar
from .region_job import NoaaRefsRegionJob, NoaaRefsSourceFileCoord
from .template_config import NoaaRefsForecastVirtualTemplateConfig


class NoaaRefsForecastVirtualDataset(
    NoaaRrfsDataset[NoaaRefsDataVar, NoaaRefsSourceFileCoord]
):
    update_offset_minutes: NonNegativeInt = 115
    update_deadline_minutes: PositiveInt = 125
    template_config: NoaaRefsForecastVirtualTemplateConfig = (
        NoaaRefsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsRegionJob] = NoaaRefsRegionJob
