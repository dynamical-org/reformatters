from collections.abc import Iterable, Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob

from .region_job import (
    UarizonaSwannAnalysisRegionJob,
    UarizonaSwannAnalysisSourceFileCoord,
)
from .template_config import UarizonaSwannAnalysisTemplateConfig, UarizonaSwannDataVar
from .validators import (
    check_data_is_current,
    check_latest_time_nans,
    check_random_time_within_last_year_nans,
)


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
        return (
            check_data_is_current,
            check_latest_time_nans,
            check_random_time_within_last_year_nans,
        )

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
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
