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
            # Data for forecast hour 0 is available 54 mins after init time on most issuances
            # We could of course increase this to hourly
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
            # Run this 23 mins after the operational update, which is at 57 */3 * * *
            # That is, at 57 + 23 = 80th min of the hour, which is 20 mins into the next hour.
            # To get every third hour, but offset by +1 hr 20 min versus the main cron,
            # we do "20 1-23/3 * * *"
            schedule="20 1-23/3 * * *",
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
