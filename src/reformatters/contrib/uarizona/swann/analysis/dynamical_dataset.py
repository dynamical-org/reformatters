from collections.abc import Iterable, Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import (
    Job,
    ReformatCronJob,
    ValidationCronJob,
)

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

_OPERATIONAL_CRON_SCHEDULE = "0 0 * * *"
_VALIDATION_CRON_SCHEDULE = "0 0 * * *"


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

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[Job]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-operational-update",
            schedule=_OPERATIONAL_CRON_SCHEDULE,
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="14G",
            shared_memory="6Gi",
            ephemeral_storage="10G",
            secret_names=[self.storage_config.k8s_secret_name],
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validation",
            schedule=_VALIDATION_CRON_SCHEDULE,
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=[self.storage_config.k8s_secret_name],
        )

        return [operational_update_cron_job, validation_cron_job]
