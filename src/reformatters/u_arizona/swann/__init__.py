from reformatters.common.dynamical_dataset import DynamicalDataset

from .region_job import SWANNRegionJob, SWANNSourceFileCoord
from .template_config import SWANNDataVar, SWANNTemplateConfig


class SWANNDataset(DynamicalDataset[SWANNDataVar, SWANNSourceFileCoord]):
    template_config: SWANNTemplateConfig = SWANNTemplateConfig()
    region_job_class: type[SWANNRegionJob] = SWANNRegionJob
