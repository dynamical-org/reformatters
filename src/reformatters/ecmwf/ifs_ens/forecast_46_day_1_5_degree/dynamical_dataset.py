import math
from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.config_models import source_fill_value_var_names
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.ecmwf.ifs_ens.forecast_46_day_config_models import (
    EcmwfIfsEns46DayDataVar,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_region_job import (
    EcmwfIfsEns46DayRegionJob,
    EcmwfIfsEns46DaySourceFileCoord,
)

from .template_config import EcmwfIfsEnsForecast46Day15DegreeTemplateConfig


class EcmwfIfsEnsForecast46Day15DegreeDataset(
    DynamicalDataset[EcmwfIfsEns46DayDataVar, EcmwfIfsEns46DaySourceFileCoord]
):
    template_config: EcmwfIfsEnsForecast46Day15DegreeTemplateConfig = (
        EcmwfIfsEnsForecast46Day15DegreeTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEns46DayRegionJob] = EcmwfIfsEns46DayRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update this dataset."""
        # Budget four jobs per worker for the latest-init redo plus one new init.
        workers = math.ceil(self.num_variable_groups() / 2)
        operational_update_cron_job = ReformatCronJob(
            name="ecmwf-ifs-ens-46-day-daily-update",
            schedule="5 10 * * *",
            suspend=False,
            concurrency_policy="Allow",
            pod_active_deadline=timedelta(hours=24),
            active_deadline=timedelta(hours=24),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="24G",
            shared_memory="6G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )
        return [
            operational_update_cron_job,
        ]

    def validators(self) -> Sequence[validation.Validator]:
        masked_vars = (
            *source_fill_value_var_names(self.template_config.data_vars),
            "pressure_level/specific_humidity",
        )
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=58, minutes=5)),
            validation.CheckRecentNans(append_dim_window=3, exclude_vars=masked_vars),
            validation.CheckRecentNans(
                append_dim_window=3,
                include_vars=masked_vars,
                max_nan_fraction=0.9999,
                spatial_sampling="quarter",
            ),
        )
