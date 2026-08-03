from collections.abc import Sequence
from datetime import timedelta
from functools import partial

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.eccc.hrdps.hrdps_config_models import EcccHrdpsDataVar
from reformatters.eccc.hrdps.region_job import EcccHrdpsSourceFileCoord

from .region_job import EcccHrdpsAnalysisRegionJob
from .template_config import EcccHrdpsAnalysisTemplateConfig


class EcccHrdpsAnalysisDataset(
    DynamicalDataset[EcccHrdpsDataVar, EcccHrdpsSourceFileCoord]
):
    template_config: EcccHrdpsAnalysisTemplateConfig = EcccHrdpsAnalysisTemplateConfig()
    region_job_class: type[EcccHrdpsAnalysisRegionJob] = EcccHrdpsAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        # HRDPS runs at 00, 06, 12, 18 UTC and each run's files are published by
        # ~init+3h49m (p99); each run extends the analysis by 6 hours.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="52 3,9,15,21 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="22G",
            shared_memory="15G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,  # remove after the initial backfill
        )

        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="22 4,10,16,22 * * *",  # 30m (pod_active_deadline) after reformat at :52
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="0.7",
            memory="3.5G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,  # remove after the initial backfill
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        # The newest analysis hour is at most ~5h old just before the next run's
        # data arrives (6h init cadence + ~4h publication lag - 5h of lead times).
        max_expected_delay = timedelta(hours=6)
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
