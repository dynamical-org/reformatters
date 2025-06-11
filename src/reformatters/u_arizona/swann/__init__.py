from collections.abc import Sequence
from functools import partial

import numpy as np

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset

from .region_job import SWANNRegionJob, SWANNSourceFileCoord
from .template_config import SWANNDataVar, SWANNTemplateConfig


class SWANNDataset(DynamicalDataset[SWANNDataVar, SWANNSourceFileCoord]):
    template_config: SWANNTemplateConfig = SWANNTemplateConfig()
    region_job_class: type[SWANNRegionJob] = SWANNRegionJob

    def validators(self) -> Sequence[validation.DataValidator]:
        # For regions outside of CONUS, the values in this dataset are expected
        # to be NaNs. We sampled various times across the dataset and determined
        # the expected number of NaNs to be 46.426% of the data.
        expected_nan_percentage = 46.425
        max_nan_percentage = expected_nan_percentage + 0.001

        most_recent_time_validator = partial(
            validation.check_analysis_recent_nans,
            max_nan_percentage=max_nan_percentage,
            sample_ds_fn=lambda ds: ds.isel(time=slice(-1, None)),
        )

        # We also want to check that the data does not have more than the expected proportion of NaNs
        # for a random time in the last year, as we pull a years worth of data for each operational
        # update of the dataset.
        random_time_index = np.random.choice(365)
        random_time_in_last_year_validator = partial(
            validation.check_analysis_recent_nans,
            max_nan_percentage=max_nan_percentage,
            sample_ds_fn=lambda ds: ds.isel(
                time=slice(random_time_index, random_time_index + 1)
            ),
        )

        return (
            most_recent_time_validator,
            random_time_in_last_year_validator,
        )
