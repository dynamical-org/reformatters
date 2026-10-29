from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
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
        """Return the kubernetes cron job definitions to operationally update this dataset."""

        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # f360 lands on S3 ~07:50 UTC and on GCS ~08:25-08:35. Leads not yet on
            # GCS fall back to S3, which throttles many concurrent pods.
            schedule="5 8 * * *",
            suspend=False,
            concurrency_policy="Allow",
            pod_active_deadline=timedelta(hours=6),
            active_deadline=timedelta(hours=6),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",
            memory="35G",
            shared_memory="19G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            # Cap parallelism to avoid 500s from overloaded s3://ecmwf-forecasts
            parallelism=min(workers, 10),
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=8, minutes=40)),
            validation.CheckRecentNans(append_dim_window=3),
        )
