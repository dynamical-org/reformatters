from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .models import NoaaRefsDataVar
from .region_job import NoaaRefsRegionJob, NoaaRefsSourceFileCoord
from .template_config import NoaaRefsForecastTemplateConfig


class NoaaRefsDataset(NoaaRrfsDataset[NoaaRefsDataVar, NoaaRefsSourceFileCoord]):
    template_config: NoaaRefsForecastTemplateConfig
    region_job_class: type[NoaaRefsRegionJob] = NoaaRefsRegionJob
