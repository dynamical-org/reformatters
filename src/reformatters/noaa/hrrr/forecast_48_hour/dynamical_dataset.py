from collections.abc import Iterable, Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import (
    CronJob,
    ReformatCronJob,
    ValidationCronJob,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord

from .region_job import NoaaHrrrForecast48HourRegionJob
from .template_config import NoaaHrrrForecast48HourTemplateConfig
from .validators import (
    check_data_is_current,
    check_forecast_completeness,
    check_forecast_recent_nans,
)


class NoaaHrrrForecast48HourDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrSourceFileCoord]
):
    """DynamicalDataset implementation for NOAA HRRR 48-hour forecast data."""

    template_config: NoaaHrrrForecast48HourTemplateConfig = (
        NoaaHrrrForecast48HourTemplateConfig()
    )
    region_job_class: type[NoaaHrrrForecast48HourRegionJob] = (
        NoaaHrrrForecast48HourRegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
        """Define Kubernetes cron jobs for operational updates and validation."""
        # We pull the 0, 6, 12, and 18 init times in this dataset
        # Update every 6 hours at 1h50m after the init time (when all forecast steps are available)
        # First file typically becomes available at 51 mins and last file (hour 48) at 1h51m
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-operational-update",
            schedule="55 1,7,13,19 * * *",
            pod_active_deadline=timedelta(minutes=15),  # usually takes 3 mins
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",
            memory="14G",
            shared_memory="400M",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        # Validation job - run 15 mins after operational update
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validation",
            schedule="10 2,8,14,20 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="0.7",
            memory="3.5G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        return (
            check_data_is_current,
            check_forecast_completeness,
            check_forecast_recent_nans,
        )
