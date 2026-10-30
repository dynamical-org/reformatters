from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.ecmwf.ifs_ens.forecast_46_day_config_models import (
    EcmwfIfsEns46DayDataVar,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_region_job import (
    EcmwfIfsEns46DayRegionJob,
    EcmwfIfsEns46DaySourceFileCoord,
)

from .template_config import (
    EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig,
)


class EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset(
    DynamicalDataset[EcmwfIfsEns46DayDataVar, EcmwfIfsEns46DaySourceFileCoord]
):
    template_config: EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig = (
        EcmwfIfsEnsForecast46Day6Hourly15DegreeTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEns46DayRegionJob] = EcmwfIfsEns46DayRegionJob
    # Validate monitor window: the trigger's validation, which follows the
    # ecmwf-ifs-ens-46-day-gribs fire, through the backstop's run.
    validation_monitor_schedule: ClassVar[str | None] = "0 3 * * *"
    validation_monitor_margin: ClassVar[timedelta | None] = timedelta(minutes=443)

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        workers = self.num_variable_groups()
        update = ReformatCronJob(
            name="ecmwf-ifs-ens-46-day-6-hourly-update",
            # Backstop schedule. Also triggered ASAP by ecmwf-ifs-ens-46-day-gribs.
            schedule="5 10 * * *",
            suspend=False,
            concurrency_policy="Allow",
            pod_active_deadline=timedelta(hours=6),
            active_deadline=timedelta(hours=6),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="12G",
            shared_memory="2.5G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )
        return [update]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=58, minutes=5)),
            validation.CheckRecentNans(append_dim_window=3),
        )
