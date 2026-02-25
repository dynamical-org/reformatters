from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

from .region_job import GefsForecast35DayRegionJob, GefsForecast35DaySourceFileCoord
from .template_config import GefsForecast35DayTemplateConfig


class GefsForecast35DayDataset(
    DynamicalDataset[GEFSDataVar, GefsForecast35DaySourceFileCoord]
):
    """GEFS 35-day forecast dataset implementation."""

    template_config: GefsForecast35DayTemplateConfig = GefsForecast35DayTemplateConfig()
    region_job_class: type[GefsForecast35DayRegionJob] = GefsForecast35DayRegionJob

    use_progress_tracker: bool = True

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="0 7 * * *",  # At 7:00 UTC every day.
            pod_active_deadline=timedelta(hours=3.5),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="6",  # fit on 8 vCPU node
            memory="120G",  # fit on 128GB node (more than needed)
            shared_memory="24G",
            ephemeral_storage="150G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="30 11 * * *",  # At 11:30 UTC every day.
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",  # fit on 4 vCPU node
            memory="30G",  # fit on 32GB node
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        return (
            validation.check_forecast_current_data,
            validation.check_forecast_recent_nans,
        )
