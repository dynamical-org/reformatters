from collections.abc import Iterable, Sequence
from datetime import timedelta

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
from .validators import (
    check_analysis_recent_nans,
    check_data_is_current,
)


class NoaaHrrrAnalysisDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrSourceFileCoord]
):
    """DynamicalDataset implementation for NOAA HRRR analysis data."""

    template_config: NoaaHrrrAnalysisTemplateConfig = NoaaHrrrAnalysisTemplateConfig()
    region_job_class: type[NoaaHrrrAnalysisRegionJob] = NoaaHrrrAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
        """Define Kubernetes cron jobs for operational updates and validation."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="55 * * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="2",
            memory="10G",
            shared_memory="400M",
            ephemeral_storage="20G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="10 * * * *",
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
            check_data_is_current,
            check_analysis_recent_nans,
        )
