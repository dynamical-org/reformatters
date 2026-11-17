from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.dwd.icon_eu.forecast_5_day.region_job import (
    DwdIconEuForecast5DaySourceFileCoord,
)
from reformatters.dwd.icon_eu.forecast_5_day.template_config import DwdIconEuDataVar

from .region_job import (
    DwdIconEuForecast5DayIcosahedralRegionJob,
)
from .template_config import DwdIconEuForecast5DayIcosahedralTemplateConfig


class DwdIconEuForecast5DayIcosahedralDataset(
    DynamicalDataset[DwdIconEuDataVar, DwdIconEuForecast5DaySourceFileCoord]
):
    template_config: DwdIconEuForecast5DayIcosahedralTemplateConfig = (
        DwdIconEuForecast5DayIcosahedralTemplateConfig()
    )
    region_job_class: type[DwdIconEuForecast5DayIcosahedralRegionJob] = (
        DwdIconEuForecast5DayIcosahedralRegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        workers = 2 * self.num_variable_groups()
        return [
            ReformatCronJob(
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
                suspend=True,
            )
        ]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=3, minutes=52)),
            validation.CheckRecentNans(),
        )
