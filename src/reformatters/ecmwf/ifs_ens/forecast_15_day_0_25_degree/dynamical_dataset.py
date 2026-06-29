from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfDataVar,
)

from .region_job import EcmwfIfsEnsForecast15Day025DegreeRegionJob
from .source_file_coord import IfsEnsSourceFileCoord
from .template_config import EcmwfIfsEnsForecast15Day025DegreeTemplateConfig


class EcmwfIfsEnsForecast15Day025DegreeDataset(
    DynamicalDataset[EcmwfDataVar, IfsEnsSourceFileCoord]
):
    template_config: EcmwfIfsEnsForecast15Day025DegreeTemplateConfig = (
        EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast15Day025DegreeRegionJob] = (
        EcmwfIfsEnsForecast15Day025DegreeRegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""

        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # Gating lead group f360 typically lands ~08:00-08:09 UTC (recent days;
            # wxopticon external-ecmwf-ifs-ens-long-aws p99 08:26); pod spin-up adds
            # a head start before late-lead downloads, so fire at 08:05.
            schedule="5 8 * * *",
            suspend=False,
            pod_active_deadline=timedelta(minutes=35),  # runs take <26 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",
            memory="35G",
            shared_memory="19G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=min(workers, 20),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="40 8 * * *",  # 35m (pod_active_deadline) after reformat at 08:05
            suspend=False,
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="0.5",
            memory="30G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        return (
            validation.check_forecast_current_data,
            validation.check_forecast_recent_nans,
        )
