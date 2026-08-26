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

from .region_job import NoaaHrrrAnalysisRegionJob
from .template_config import NoaaHrrrAnalysisTemplateConfig


class NoaaHrrrAnalysisDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrSourceFileCoord]
):
    """DynamicalDataset implementation for NOAA HRRR analysis."""

    template_config: NoaaHrrrAnalysisTemplateConfig = NoaaHrrrAnalysisTemplateConfig()
    region_job_class: type[NoaaHrrrAnalysisRegionJob] = NoaaHrrrAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Define Kubernetes cron jobs for operational updates and validation."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.cron_job_name_prefix}-update",
            # Every 3 hours at 57 minutes past the hour.
            # HRRR f001 (last lead time used) NOMADS last-modified ~init+53m (max ~55m;
            # we try S3 first to spare NOMADS, but NOMADS publishes first, by ~10 min at
            # 06z). We could of course increase this to hourly.
            schedule="57 */3 * * *",
            pod_active_deadline=timedelta(minutes=20),  # runs take <15 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="45G",
            shared_memory="16.5G",
            ephemeral_storage="60G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        validation_cron_job = ValidationCronJob(
            name=f"{self.cron_job_name_prefix}-validate",
            # 20m (pod_active_deadline) after reformat at :57 = :77 = :17 of the next hour.
            # "17 1-23/3 * * *" gives 01:17, 04:17, 07:17, ... matching reformat at 00:57, 03:57, 06:57, ...
            schedule="17 1-23/3 * * *",
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
            validation.CheckCurrentData(max_delay=timedelta(hours=4)),
            validation.CheckRecentNans(exclude_vars=source_fill_value_vars),
            # NaN here is the source's no-precipitation / no-cloud-ceiling marker, so
            # coverage is small and clustered: a sampled quadrant is regularly all
            # marker, while whole-grid coverage stays well clear of the threshold
            # (percent frozen precipitation, the sparser of the two, peaks near 0.996
            # in the driest hours of the year).
            validation.CheckRecentNans(
                include_vars=source_fill_value_vars,
                max_nan_fraction=0.999,
                spatial_sampling="all",
                append_dim_window=4,
            ),
        )
