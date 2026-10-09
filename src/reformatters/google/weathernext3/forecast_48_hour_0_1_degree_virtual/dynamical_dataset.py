from pydantic import Field

from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.google.weathernext3.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext3ForecastVirtualDataset,
)
from reformatters.google.weathernext_virtual.listing import (
    weathernext_virtual_chunk_containers,
)

from .region_job import GoogleWeathernext3Forecast48Hour01DegreeVirtualRegionJob
from .template_config import (
    GoogleWeathernext3Forecast48Hour01DegreeVirtualTemplateConfig,
)


class GoogleWeathernext3Forecast48Hour01DegreeVirtualDataset(
    GoogleWeathernext3ForecastVirtualDataset
):
    template_config: GoogleWeathernext3Forecast48Hour01DegreeVirtualTemplateConfig = (
        GoogleWeathernext3Forecast48Hour01DegreeVirtualTemplateConfig()
    )
    region_job_class: type[GoogleWeathernext3Forecast48Hour01DegreeVirtualRegionJob] = (
        GoogleWeathernext3Forecast48Hour01DegreeVirtualRegionJob
    )
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=weathernext_virtual_chunk_containers(),
            manifest_split=manifest_append_dim_split(split_size=384, dim="init_time"),
        )
    )
