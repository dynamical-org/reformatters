from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.dwd.icon_eu.icon_eu_template_config import DwdIconEuDataVar

from .region_job import (
    DwdIconEuForecast5DayIcosahedralRegionJob,
    DwdIconEuForecast5DayIcosahedralSourceFileCoord,
)
from .template_config import DwdIconEuForecast5DayIcosahedralTemplateConfig

# Hand-set from 4 complete native-grid runs over one day (last file at most +3h14m after init);
# DWD retains about a day of runs, so the sample is small. Re-derive from the timing profiler after ~30 live runs.
UPDATE_SCHEDULE = "17 3,9,15,21 * * *"
UPDATE_MAX_DELAY = timedelta(hours=3, minutes=17)
UPDATE_DEADLINE = timedelta(hours=6)


class DwdIconEuForecast5DayIcosahedralDataset(
    DynamicalDataset[DwdIconEuDataVar, DwdIconEuForecast5DayIcosahedralSourceFileCoord]
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
                schedule=UPDATE_SCHEDULE,
                concurrency_policy="Allow",
                pod_active_deadline=UPDATE_DEADLINE,
                active_deadline=UPDATE_DEADLINE,
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
            validation.CheckCurrentData(max_delay=UPDATE_MAX_DELAY),
            validation.CheckRecentNans(),
        )
