from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.config_models import source_fill_value_var_names
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

    def validators(self) -> Sequence[validation.Validator]:
        # Source-fill-value vars are NaN wherever the source's missing state applies
        # (e.g. percent frozen precipitation where nothing is falling), so they get a
        # looser check: not entirely NaN.
        source_fill_value_vars = source_fill_value_var_names(
            self.template_config.data_vars
        )
        return (
            validation.CheckCurrentData(max_age=timedelta(hours=7)),
            # window=4 covers a day of 6-hourly cycles, so a truncated or missing
            # forecast is caught even after newer cycles land.
            validation.CheckRecentNans(exclude_vars=source_fill_value_vars, window=4),
            validation.CheckRecentNans(
                include_vars=source_fill_value_vars,
                max_nan_fraction=0.9999,
                spatial_sampling="quarter",
            ),
        )
