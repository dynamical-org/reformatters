from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar

from .region_job import (
    EcmwfAifsSingleForecastRegionJob,
    EcmwfAifsSingleForecastSourceFileCoord,
)
from .template_config import EcmwfAifsSingleForecastTemplateConfig


class EcmwfAifsSingleForecastDataset(
    DynamicalDataset[EcmwfDataVar, EcmwfAifsSingleForecastSourceFileCoord]
):
    template_config: EcmwfAifsSingleForecastTemplateConfig = (
        EcmwfAifsSingleForecastTemplateConfig()
    )
    region_job_class: type[EcmwfAifsSingleForecastRegionJob] = (
        EcmwfAifsSingleForecastRegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # Source GRIB/index pairs normally finish near init+7h35m.
            schedule="45 1,7,13,19 * * *",
            pod_active_deadline=timedelta(minutes=10),  # runs take ~1 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            memory="7G",
            shared_memory="1.5G",
            ephemeral_storage="20G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=7, minutes=55)),
            validation.CheckRecentNans(append_dim_window=3),
        )
