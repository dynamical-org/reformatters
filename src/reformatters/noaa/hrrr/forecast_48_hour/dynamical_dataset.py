from collections.abc import Sequence
from datetime import timedelta
from functools import partial

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
from .validators import HRRR_EXPECTED_HOUR_0_NAN_VARS, check_forecast_completeness


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

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Define Kubernetes cron jobs for operational updates and validation."""
        # We pull the 0, 6, 12, and 18 init times in this dataset.
        # HRRR f048 (last lead time) NOMADS last-modified ~init+1h50m (we try S3
        # first to spare NOMADS, but NOMADS publishes first). +3 min buffer.
        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="53 1,7,13,19 * * *",
            pod_active_deadline=timedelta(minutes=10),  # usually takes <2 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",
            memory="14G",
            shared_memory="400M",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )

        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="3 2,8,14,20 * * *",  # 10m (pod_active_deadline) after reformat at :53
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
            partial(
                validation.check_forecast_current_data,
                max_latest_init_time_age=timedelta(hours=7),
            ),
            check_forecast_completeness,
            partial(
                validation.check_forecast_recent_nans,
                additional_skip_lead_time_0_vars=HRRR_EXPECTED_HOUR_0_NAN_VARS,
                # CF-masks its -50 "no precipitation" sentinel to NaN, so the field
                # is legitimately all/mostly NaN wherever no precipitation is falling.
                exclude_vars=("percent_frozen_precipitation_surface",),
            ),
        )
