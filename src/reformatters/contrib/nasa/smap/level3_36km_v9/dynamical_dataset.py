from collections.abc import Sequence
from datetime import timedelta
from functools import partial

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob

from .region_job import (
    NasaSmapLevel336KmV9RegionJob,
    NasaSmapLevel336KmV9SourceFileCoord,
)
from .template_config import NasaSmapDataVar, NasaSmapLevel336KmV9TemplateConfig


class NasaSmapLevel336KmV9Dataset(
    DynamicalDataset[NasaSmapDataVar, NasaSmapLevel336KmV9SourceFileCoord]
):
    template_config: NasaSmapLevel336KmV9TemplateConfig = (
        NasaSmapLevel336KmV9TemplateConfig()
    )
    region_job_class: type[NasaSmapLevel336KmV9RegionJob] = (
        NasaSmapLevel336KmV9RegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-operational-update",
            # New data comes 1x per day, usually around 5:45 but sometimes later, more like 14:00
            # Run twice to pick up the later data too (updates only take a couple minutes)
            schedule="0 6,18 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",
            memory="7G",
            shared_memory="600M",
            ephemeral_storage="20G",
            # Earthdata credentials required to download source data
            secret_names=[*self.store_factory.k8s_secret_names(), "nasa-earthdata"],
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validation",
            schedule="30 6,18 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        # Usually < 24 hours, but this isn't super latency sensitive data
        max_expected_delay = timedelta(hours=48)
        return (
            partial(
                validation.check_analysis_current_data,
                max_expected_delay=max_expected_delay,
            ),
            partial(
                validation.check_analysis_recent_nans,
                max_expected_delay=max_expected_delay,
                # Oceans and about half of land (due to swaths) are expected to be NaNs
                max_nan_percentage=90,
            ),
        )
