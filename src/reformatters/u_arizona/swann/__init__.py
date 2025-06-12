from collections.abc import Sequence

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset

from .region_job import SWANNRegionJob, SWANNSourceFileCoord
from .template_config import SWANNDataVar, SWANNTemplateConfig
from .validators import (
    check_data_is_current,
    check_latest_time_nans,
    check_random_time_within_last_year_nans,
)


class SWANNDataset(DynamicalDataset[SWANNDataVar, SWANNSourceFileCoord]):
    template_config: SWANNTemplateConfig = SWANNTemplateConfig()
    region_job_class: type[SWANNRegionJob] = SWANNRegionJob

    def validators(self) -> Sequence[validation.DataValidator]:
        return (
            check_data_is_current,
            check_latest_time_nans,
            check_random_time_within_last_year_nans,
        )
