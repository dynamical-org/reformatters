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
from reformatters.noaa.gfs.region_job import NoaaGfsSourceFileCoord
from reformatters.noaa.models import NoaaDataVar

from .region_job import NoaaGfsAnalysisRegionJob
from .template_config import NoaaGfsAnalysisTemplateConfig


class NoaaGfsAnalysisDataset(DynamicalDataset[NoaaDataVar, NoaaGfsSourceFileCoord]):
    """DynamicalDataset implementation for NOAA GFS analysis."""

    template_config: NoaaGfsAnalysisTemplateConfig = NoaaGfsAnalysisTemplateConfig()
    region_job_class: type[NoaaGfsAnalysisRegionJob] = NoaaGfsAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
        """Define Kubernetes cron jobs for operational updates and validation."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="30 */6 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            memory="7G",
            shared_memory="1.5G",
            ephemeral_storage="20G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="0 1,7,13,19 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        max_expected_delay = timedelta(hours=7)
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
