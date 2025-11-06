from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfDataVar,
)

from .region_job import (
    EcmwfIfsEnsForecast15Day025DegreeRegionJob,
    EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord,
)
from .template_config import EcmwfIfsEnsForecast15Day025DegreeTemplateConfig


class EcmwfIfsEnsForecast15Day025DegreeDataset(
    DynamicalDataset[EcmwfDataVar, EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]
):
    template_config: EcmwfIfsEnsForecast15Day025DegreeTemplateConfig = (
        EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast15Day025DegreeRegionJob] = (
        EcmwfIfsEnsForecast15Day025DegreeRegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""

        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # ECMWF uploads the first file at 07:40 UTC and the last one by ~07:45 UTC.
            # (Ensemble stats get uploaded 15-20 mins later, but we don't process those.)
            schedule="50 7 * * *",
            suspend=True,
            # Temporarily increase deadline while doing catchup run
            pod_active_deadline=timedelta(hours=12),
            # pod_active_deadline=timedelta(minutes=45),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",
            memory="35G",
            shared_memory="19G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validation",
            schedule="40 8 * * *",  # 50 minutes after update starts
            suspend=True,
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="0.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        return (
            validation.check_forecast_current_data,
            validation.check_forecast_recent_nans,
        )
