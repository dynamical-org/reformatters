from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from reformatters.common import validation
from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisMaterializedRegionJob,
    UcsbChcChirpsAnalysisSourceFileCoord,
)


class UcsbChcChirpsAnalysisMaterializedDataset(
    DynamicalDataset[DataVar[BaseInternalAttrs], UcsbChcChirpsAnalysisSourceFileCoord]
):
    region_job_class: type[UcsbChcChirpsAnalysisMaterializedRegionJob]

    update_schedule: ClassVar[str]

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule=self.update_schedule,
            pod_active_deadline=timedelta(minutes=60),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            memory="45G",
            shared_memory="25.5G",
            ephemeral_storage="20G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(
                max_delay=self.region_job_class.expected_unavailable_window
            ),
            validation.CheckRecentNans(
                # Points that are NaN throughout the sampled window are excluded.
                max_nan_fraction=0.05,
                sampled_points=40,
                append_dim_window=3,
            ),
        )
