from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar

from .region_job import (
    EcmwfAifsEnsForecastRegionJob,
    EcmwfAifsEnsForecastSourceFileCoord,
)
from .template_config import EcmwfAifsEnsForecastTemplateConfig


class EcmwfAifsEnsForecastDataset(
    DynamicalDataset[EcmwfDataVar, EcmwfAifsEnsForecastSourceFileCoord]
):
    template_config: EcmwfAifsEnsForecastTemplateConfig = (
        EcmwfAifsEnsForecastTemplateConfig()
    )
    region_job_class: type[EcmwfAifsEnsForecastRegionJob] = (
        EcmwfAifsEnsForecastRegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # AIFS-ENS publishes the last file (step 360h) at ~H+5h44m after init.
            # Run 3 minutes later at H+5h47m.
            schedule="47 5/6 * * *",
            pod_active_deadline=timedelta(minutes=30),
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
            # Validation runs 30 minutes after each update run: 00:17, 06:17, 12:17, and 18:17.
            schedule="17 */6 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="0.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        return (
            validation.check_forecast_current_data,
            validation.check_forecast_recent_nans,
        )
