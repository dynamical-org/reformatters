from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob

from .region_job import (
    UarizonaSwannAnalysisRegionJob,
    UarizonaSwannAnalysisSourceFileCoord,
)
from .template_config import UarizonaSwannAnalysisTemplateConfig, UarizonaSwannDataVar

# For regions outside of CONUS, the values in this dataset are expected
# to be NaNs. We sampled various times across the dataset and determined
# the expected fraction of NaNs to be ~0.46425.
EXPECTED_NAN_FRACTION = 0.46425
MAX_NAN_FRACTION = EXPECTED_NAN_FRACTION + 0.00001


class UarizonaSwannAnalysisDataset(
    DynamicalDataset[UarizonaSwannDataVar, UarizonaSwannAnalysisSourceFileCoord]
):
    template_config: UarizonaSwannAnalysisTemplateConfig = (
        UarizonaSwannAnalysisTemplateConfig()
    )
    region_job_class: type[UarizonaSwannAnalysisRegionJob] = (
        UarizonaSwannAnalysisRegionJob
    )

    def validators(self) -> Sequence[validation.Validator]:
        return (
            # SWANN data is usually published daily with just over a day lag.
            # There are occasional longer lags, allow them without alerting because
            # this is a contrib dataset.
            validation.CheckCurrentData(max_delay=timedelta(days=5)),
            validation.CheckRecentNans(
                # Check the full grid for a stable NaN fraction. Whole-grid reads
                # are per position, so this covers the newest few; the deep check
                # below follows the update over the whole year it rewrites.
                max_nan_fraction=MAX_NAN_FRACTION,
                spatial_sampling="all",
                window=2,
            ),
            validation.CheckRecentNans(
                # The update rewrites a year of data (UArizona restates files as
                # they go early -> provisional -> stable) and the window follows it.
                # Point sampling reads the whole year in one pass: a point spans 2 time
                # chunks, so this costs about what a single position does. Outside
                # CONUS every position is NaN, so those points are dropped as
                # structural holes; sample extra to be sure some land in CONUS.
                max_nan_fraction=0.0,
                sampled_points=8,
            ),
        )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="0 20 * * *",
            pod_active_deadline=timedelta(minutes=10),  # runs take <4 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="14G",
            shared_memory="6Gi",
            ephemeral_storage="10G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="10 20 * * *",  # 10m (pod_active_deadline) after reformat at :00
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]
