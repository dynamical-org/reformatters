from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.dwd.icon_eu.icon_eu_template_config import DwdIconEuDataVar

from .region_job import (
    DwdIconEuForecast5DayRegionJob,
    DwdIconEuForecast5DaySourceFileCoord,
)
from .template_config import DwdIconEuForecast5DayTemplateConfig


class DwdIconEuForecast5DayDataset(
    DynamicalDataset[DwdIconEuDataVar, DwdIconEuForecast5DaySourceFileCoord]
):
    template_config: DwdIconEuForecast5DayTemplateConfig = (
        DwdIconEuForecast5DayTemplateConfig()
    )
    region_job_class: type[DwdIconEuForecast5DayRegionJob] = (
        DwdIconEuForecast5DayRegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update this dataset."""
        # ICON-EU runs at 00, 06, 12, 18 UTC. DWD's complete forecast (f120) is available by
        # ~init+3h49m (p99). We schedule the reformat 3 minutes after that (3h52m after init).
        # The archive job may not have finished copying to Source Coop yet, in which case
        # download_file falls back to reading directly from DWD.
        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="52 3,9,15,21 * * *",
            concurrency_policy="Allow",
            pod_active_deadline=timedelta(hours=6),
            active_deadline=timedelta(hours=6),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",
            memory="14G",
            shared_memory="400M",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=3, minutes=52)),
            validation.CheckRecentNans(),
        )
