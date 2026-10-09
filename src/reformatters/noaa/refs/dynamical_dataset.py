from collections.abc import Sequence

from pydantic import NonNegativeInt, PositiveInt

from reformatters.common.kubernetes import CronJob
from reformatters.common.pydantic import replace
from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .models import NoaaRefsDataVar
from .region_job import NoaaRefsRegionJob, NoaaRefsSourceFileCoord
from .template_config import NoaaRefsForecastTemplateConfig


class NoaaRefsDataset(NoaaRrfsDataset[NoaaRefsDataVar, NoaaRefsSourceFileCoord]):
    update_offset_minutes: NonNegativeInt = 115
    update_deadline_minutes: PositiveInt = 125
    template_config: NoaaRefsForecastTemplateConfig
    region_job_class: type[NoaaRefsRegionJob] = NoaaRefsRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return tuple(
            replace(job, suspend=True)
            for job in super().operational_kubernetes_resources(image_tag)
        )
