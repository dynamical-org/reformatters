from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

from .region_job import GefsAnalysisRegionJob, GefsAnalysisSourceFileCoord
from .template_config import GefsAnalysisTemplateConfig


class GefsAnalysisDataset(DynamicalDataset[GEFSDataVar, GefsAnalysisSourceFileCoord]):
    """GEFS analysis dataset implementation."""

    template_config: GefsAnalysisTemplateConfig = GefsAnalysisTemplateConfig()
    region_job_class: type[GefsAnalysisRegionJob] = GefsAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # GEFS f006 (last lead time used) available ~3h48m after init on NOMADS. +3 min buffer.
            schedule="51 3,9,15,21 * * *",
            pod_active_deadline=timedelta(hours=1),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="14",  # fit on 16 vCPU node
            memory="30G",  # fit on 32GB node
            shared_memory="12G",
            ephemeral_storage="35G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="51 4,10,16,22 * * *",  # 1h (pod_active_deadline) after reformat
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",  # fit on 2 vCPU node
            memory="7G",  # fit on 8GB node
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        return (
            validation.check_analysis_current_data,
            validation.check_analysis_recent_nans,
        )
