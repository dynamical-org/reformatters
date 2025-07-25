from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob

from .region_job import NoaaNdviCdrAnalysisRegionJob, NoaaNdviCdrAnalysisSourceFileCoord
from .template_config import NoaaNdviCdrAnalysisTemplateConfig, NoaaNdviCdrDataVar
from .validators import check_data_is_current, check_latest_ndvi_usable_nan_percentage


class NoaaNdviCdrAnalysisDataset(
    DynamicalDataset[NoaaNdviCdrDataVar, NoaaNdviCdrAnalysisSourceFileCoord]
):
    template_config: NoaaNdviCdrAnalysisTemplateConfig = (
        NoaaNdviCdrAnalysisTemplateConfig()
    )
    region_job_class: type[NoaaNdviCdrAnalysisRegionJob] = NoaaNdviCdrAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-operational-update",
            schedule="0 20 * * *",
            pod_active_deadline=timedelta(minutes=60),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="2",
            memory="100G",
            shared_memory="76Gi",
            ephemeral_storage="150G",
            secret_names=self.storage_config.k8s_secret_names,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validation",
            schedule="30 21 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.storage_config.k8s_secret_names,
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        return (
            check_data_is_current,
            check_latest_ndvi_usable_nan_percentage,
        )
