from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar

from .region_job import EcmwfAifsForecastRegionJob, EcmwfAifsForecastSourceFileCoord
from .template_config import EcmwfAifsForecastTemplateConfig


class EcmwfAifsForecastDataset(
    DynamicalDataset[EcmwfDataVar, EcmwfAifsForecastSourceFileCoord]
):
    template_config: EcmwfAifsForecastTemplateConfig = EcmwfAifsForecastTemplateConfig()
    region_job_class: type[EcmwfAifsForecastRegionJob] = EcmwfAifsForecastRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        suspend = True
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="25 /6 * * *",
            suspend=suspend,
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
            name=f"{self.dataset_id}-validate",
            schedule="55 /6 * * *",
            suspend=suspend,
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        return (
            validation.check_forecast_current_data,
            validation.check_forecast_recent_nans,
        )
