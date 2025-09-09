from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

from .region_job import GefsAnalysisRegionJob
from .source_file_coord import GefsAnalysisSourceFileCoord
from .template_config import GefsAnalysisTemplateConfig

# Operational and validation cron schedules from existing implementation
_OPERATIONAL_CRON_SCHEDULE = "0 0,6,12,18 * * *"  # UTC
_VALIDATION_CRON_SCHEDULE = "30 7,10,13,19 * * *"  # UTC 1.5 hours after update


class GefsAnalysisDataset(DynamicalDataset[GEFSDataVar, GefsAnalysisSourceFileCoord]):
    """GEFS analysis dataset implementation."""

    template_config: GefsAnalysisTemplateConfig = GefsAnalysisTemplateConfig()
    region_job_class: type[GefsAnalysisRegionJob] = GefsAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset.

        Based on existing operational_kubernetes_resources() function in analysis/reformat.py
        """
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-operational-update",
            schedule=_OPERATIONAL_CRON_SCHEDULE,
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
            name=f"{self.dataset_id}-validation",
            schedule=_VALIDATION_CRON_SCHEDULE,
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",  # fit on 2 vCPU node
            memory="7G",  # fit on 8GB node
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset.

        Based on existing validate_dataset() function in analysis/reformat.py
        """
        return (
            validation.check_analysis_current_data,
            validation.check_analysis_recent_nans,
        )
