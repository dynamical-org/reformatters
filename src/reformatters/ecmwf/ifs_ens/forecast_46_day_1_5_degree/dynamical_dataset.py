from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.config_models import source_fill_value_var_names
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.ecmwf.ifs_ens.s2s_dynamical_dataset import (
    ECDS_API_KEY_SECRET_NAME,
    SOURCE_COOP_SECRET_NAME,
    EcmwfS2sDynamicalDataset,
)
from reformatters.ecmwf.ifs_ens.s2s_region_job import EcmwfS2sRegionJob

from .template_config import EcmwfIfsEnsForecast46Day15DegreeTemplateConfig


class EcmwfIfsEnsForecast46Day15DegreeDataset(EcmwfS2sDynamicalDataset):
    template_config: EcmwfIfsEnsForecast46Day15DegreeTemplateConfig = (
        EcmwfIfsEnsForecast46Day15DegreeTemplateConfig()
    )
    region_job_class: type[EcmwfS2sRegionJob] = EcmwfS2sRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        # Archiving GRIBs does not read the store, so the archive job runs before the
        # backfill that creates one.
        reformat_suspend = (
            True  # Remove after backfilling to run operational updates and validation
        )
        archive_grib_files_job = CronJob(
            command=["archive-grib-files"],
            workers_total=1,
            parallelism=1,
            name=f"{self.dataset_id}-archive-grib-files",
            schedule="0 6 * * *",
            pod_active_deadline=timedelta(hours=6),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="8G",
            ephemeral_storage="60G",
            secret_names=[SOURCE_COOP_SECRET_NAME, ECDS_API_KEY_SECRET_NAME],
        )
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
            parallelism=min(workers, 10),
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
            archive_grib_files_job,
            operational_update_cron_job,
            validation_cron_job,
        ]

    def validators(self) -> Sequence[validation.Validator]:
        # NaN by construction, at a fraction that varies with where the sampled points
        # land. Whole-grid sampling would make it stable but reads every lead time of
        # all 101 members, so they are gated on completeness by CheckExpectedShards
        # instead.
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
        )
