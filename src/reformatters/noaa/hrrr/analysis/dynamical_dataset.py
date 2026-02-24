from collections.abc import Iterable, Sequence
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

from .region_job import NoaaHrrrAnalysisRegionJob
from .template_config import NoaaHrrrAnalysisTemplateConfig


class NoaaHrrrAnalysisDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrSourceFileCoord]
):
    """DynamicalDataset implementation for NOAA HRRR analysis."""

    template_config: NoaaHrrrAnalysisTemplateConfig = NoaaHrrrAnalysisTemplateConfig()
    region_job_class: type[NoaaHrrrAnalysisRegionJob] = NoaaHrrrAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
        """Define Kubernetes cron jobs for operational updates and validation."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # Every 3 hours at 57 minutes past the hour.
            # HRRR f001 (last lead time used) available ~54m after init on NOMADS. +3 min buffer.
            # We could of course increase this to hourly.
            schedule="57 */3 * * *",
            pod_active_deadline=timedelta(minutes=20),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="45G",
            shared_memory="16.5G",
            ephemeral_storage="60G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
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

    def validators(self) -> Sequence[validation.DataValidator]:
        max_expected_delay = timedelta(hours=4)
        return (
            partial(
                validation.check_analysis_current_data,
                max_expected_delay=max_expected_delay,
            ),
            partial(
                validation.check_analysis_recent_nans,
                max_expected_delay=max_expected_delay,
            ),
        )
