from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.config_models import source_fill_value_var_names
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
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
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        reformat_suspend = True
        workers = self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="0 9 * * *",
            suspend=reformat_suspend,
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
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="0 12 * * *",
            suspend=reformat_suspend,
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1",
            memory="30G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        return [
            operational_update_cron_job,
            validation_cron_job,
        ]

    def validators(self) -> Sequence[validation.Validator]:
        masked_vars = (
            *source_fill_value_var_names(self.template_config.data_vars),
            "pressure_level/specific_humidity",
        )
        return (
            # ECDS publishes about 53 hours after the reference time and the update
            # runs at 09 UTC, so an initialization lands about 57 hours out. Four days
            # leaves room for one missed cycle, which the next day's update fills.
            validation.CheckCurrentData(max_delay=timedelta(days=4)),
            validation.CheckRecentNans(window=3, exclude_vars=masked_vars),
            validation.CheckRecentNans(
                window=3,
                include_vars=masked_vars,
                max_nan_fraction=0.9999,
                spatial_sampling="quarter",
            ),
        )
