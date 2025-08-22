from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.noaa.models import NoaaDataVar

from .region_job import NoaaGfsForecastRegionJob, NoaaGfsForecastSourceFileCoord
from .template_config import NoaaGfsForecastTemplateConfig


class NoaaGfsForecastDataset(
    DynamicalDataset[NoaaDataVar, NoaaGfsForecastSourceFileCoord]
):
    template_config: NoaaGfsForecastTemplateConfig = NoaaGfsForecastTemplateConfig()
    region_job_class: type[NoaaGfsForecastRegionJob] = NoaaGfsForecastRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-operational-update",
            schedule="30 5,11,17,23 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            memory="7G",
            shared_memory="1.5G",
            ephemeral_storage="20G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validation",
            schedule="0 6,12,18,0 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """The sequence of DataValidators to run on this dataset."""
        return (
            validation.check_forecast_current_data,
            validation.check_forecast_recent_nans,
        )
