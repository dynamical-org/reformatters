from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
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

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        workers = self.num_variable_groups()
        update = ReformatCronJob(
            name="ecmwf-ifs-ens-46-day-6-hourly-update",
            schedule="0 10 * * *",
            suspend=True,
            pod_active_deadline=timedelta(hours=3),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="30G",
            shared_memory="12G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )
        validate = ValidationCronJob(
            name="ecmwf-ifs-ens-46-day-6-hourly-validate",
            schedule="0 13 * * *",
            suspend=True,
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1",
            memory="30G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        return [update, validate]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(days=4)),
            validation.CheckRecentNans(append_dim_window=3),
        )
