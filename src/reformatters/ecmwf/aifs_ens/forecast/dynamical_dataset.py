from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
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
            # AIFS-ENS full completion (wxopticon external-ecmwf-aifs-ens-aws, 365d):
            # last file p50 H+5h56m, p95 H+7h11m. Fire at H+7h00m (01/07/13/19 UTC);
            # late leads download last, so they land before the run reaches them.
            # The rare p99 (H+13h+) tail is left to validation-failure reruns.
            schedule="0 1,7,13,19 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="30G",
            shared_memory="13G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=min(workers, 20),
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=7, minutes=30)),
            validation.CheckRecentNans(append_dim_window=3),
        )
