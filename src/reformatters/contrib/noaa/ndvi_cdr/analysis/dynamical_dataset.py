from collections.abc import Sequence
from datetime import timedelta
from functools import partial

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob

from .region_job import NoaaNdviCdrAnalysisRegionJob, NoaaNdviCdrAnalysisSourceFileCoord
from .template_config import NoaaNdviCdrAnalysisTemplateConfig, NoaaNdviCdrDataVar


class NoaaNdviCdrAnalysisDataset(
    DynamicalDataset[NoaaNdviCdrDataVar, NoaaNdviCdrAnalysisSourceFileCoord]
):
    template_config: NoaaNdviCdrAnalysisTemplateConfig = (
        NoaaNdviCdrAnalysisTemplateConfig()
    )
    region_job_class: type[NoaaNdviCdrAnalysisRegionJob] = NoaaNdviCdrAnalysisRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="0 20 * * *",
            pod_active_deadline=timedelta(minutes=30),  # runs take <24 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="2",
            memory="100G",
            shared_memory="76Gi",
            ephemeral_storage="150G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="30 20 * * *",  # 30m (pod_active_deadline) after reformat at :00
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        # There's usually a ~3 day lag for this data's availability, occasionally much longer.
        max_expected_delay = timedelta(days=30)
        return (
            partial(
                validation.check_analysis_current_data,
                max_expected_delay=max_expected_delay,
            ),
            partial(
                validation.check_analysis_recent_nans,
                max_expected_delay=max_expected_delay,
                # Large NaN fraction is expected: oceans and water bodies are always NaN
                # (~93% baseline, observed up to ~96%). Use full-grid sampling because
                # structural NaN makes random_points bimodal/unstable.
                max_nan_fraction=0.97,
                include_vars=["ndvi_usable"],
                spatial_sampling="all",
            ),
        )
