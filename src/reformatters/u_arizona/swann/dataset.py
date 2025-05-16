from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.region_job import RegionJob
from reformatters.common.template_config import TemplateConfig

from .region_job import SWANNRegionJob, SWANNSourceFileCoord
from .template_config import SWANNDataVar, SWANNTemplateConfig


class SWANNDataset(DynamicalDataset[SWANNDataVar, SWANNSourceFileCoord]):
    template_config: TemplateConfig[SWANNDataVar] = SWANNTemplateConfig()
    region_job_class: type[RegionJob[SWANNDataVar, SWANNSourceFileCoord]] = (
        SWANNRegionJob
    )
