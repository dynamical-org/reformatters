from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsDataVar

from .region_job import GefsAnalysisRegionJob, GefsAnalysisSourceFileCoord
from .template_config import GefsAnalysisTemplateConfig


class GefsAnalysisDataset(
    DynamicalDataset[NoaaGefsDataVar, GefsAnalysisSourceFileCoord]
):
    """GEFS analysis dataset implementation."""

    template_config: GefsAnalysisTemplateConfig = GefsAnalysisTemplateConfig()
    region_job_class: type[GefsAnalysisRegionJob] = GefsAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update this dataset."""
        # GEFS analysis overrides max_vars_per_job to 1 for logic simplicity.
        # Using num_variable_groups() would create way more parallelism than is useful
        workers = 2
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="8 4,10,16,22 * * *",
            concurrency_policy="Allow",
            pod_active_deadline=timedelta(hours=6),
            active_deadline=timedelta(hours=6),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="14",  # fit on 16 vCPU node
            memory="30G",  # fit on 32GB node
            shared_memory="12G",
            ephemeral_storage="35G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=4, minutes=8)),
            validation.CheckRecentNans(),
        )
