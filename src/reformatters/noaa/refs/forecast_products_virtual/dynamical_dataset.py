from typing import ClassVar

from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

from .models import NoaaRefsProductsDataVar
from .region_job import NoaaRefsProductsRegionJob, NoaaRefsProductsSourceFileCoord
from .template_config import NoaaRefsForecastProductsVirtualTemplateConfig


class NoaaRefsForecastProductsVirtualDataset(
    NoaaRrfsDataset[NoaaRefsProductsDataVar, NoaaRefsProductsSourceFileCoord]
):
    _operational_timing: ClassVar[tuple[int, int]] = (115, 125)
    template_config: NoaaRefsForecastProductsVirtualTemplateConfig = (
        NoaaRefsForecastProductsVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRefsProductsRegionJob] = NoaaRefsProductsRegionJob
