from collections.abc import Sequence
from datetime import timedelta
from functools import partial

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob

from .region_job import (
    UarizonaSwannAnalysisRegionJob,
    UarizonaSwannAnalysisSourceFileCoord,
)
from .template_config import UarizonaSwannAnalysisTemplateConfig, UarizonaSwannDataVar
from .validators import MAX_NAN_FRACTION, check_random_time_within_last_year_nans


class UarizonaSwannAnalysisDataset(
    DynamicalDataset[UarizonaSwannDataVar, UarizonaSwannAnalysisSourceFileCoord]
):
    template_config: UarizonaSwannAnalysisTemplateConfig = (
        UarizonaSwannAnalysisTemplateConfig()
    )
    region_job_class: type[UarizonaSwannAnalysisRegionJob] = (
        UarizonaSwannAnalysisRegionJob
    )

    def validators(self) -> Sequence[validation.DataValidator]:
        # SWANN data is published daily with a few-day lag.
        max_expected_delay = timedelta(days=5)
        return (
            partial(
                validation.check_analysis_current_data,
                max_expected_delay=max_expected_delay,
            ),
            partial(
                validation.check_analysis_recent_nans,
                max_expected_delay=max_expected_delay,
                # SWANN's CONUS grid has ~46.4% structural NaN (Mexico/Canada/coast
                # within the bounding box). With random_points sampling that gives
                # a bimodal {0, 0.5, 1.0} per-run fraction; use full-grid sampling
                # since the dataset is small (~35MB for 5 days x 2 vars).
                max_nan_fraction=MAX_NAN_FRACTION,
                sampling_strategy="all",
            ),
            check_random_time_within_last_year_nans,
        )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="0 20 * * *",
            pod_active_deadline=timedelta(minutes=30),
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
            schedule="30 20 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]
