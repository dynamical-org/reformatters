from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import (
    CronJob,
    ReformatCronJob,
)
from reformatters.noaa.gfs.region_job import NoaaGfsSourceFileCoord
from reformatters.noaa.models import NoaaDataVar

from .region_job import NoaaGfsAnalysisRegionJob
from .template_config import NoaaGfsAnalysisTemplateConfig


class NoaaGfsAnalysisDataset(DynamicalDataset[NoaaDataVar, NoaaGfsSourceFileCoord]):
    """DynamicalDataset implementation for NOAA GFS analysis."""

    template_config: NoaaGfsAnalysisTemplateConfig = NoaaGfsAnalysisTemplateConfig()
    region_job_class: type[NoaaGfsAnalysisRegionJob] = NoaaGfsAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Define Kubernetes cron jobs for operational updates."""
        # GFS f006 (last lead time used) NOMADS last-modified ~init+3h36m. +3 min buffer.
        workers = self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="40 3,9,15,21 * * *",
            pod_active_deadline=timedelta(minutes=10),  # runs take <3 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="40G",
            shared_memory="13.5G",
            ephemeral_storage="50G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=7)),
            validation.CheckRecentNans(),
        )
