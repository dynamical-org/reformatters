from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from reformatters.common import validation
from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisMaterializedRegionJob,
    UcsbChcChirpsAnalysisSourceFileCoord,
)


class UcsbChcChirpsAnalysisMaterializedDataset(
    DynamicalDataset[DataVar[BaseInternalAttrs], UcsbChcChirpsAnalysisSourceFileCoord]
):
    """Shared base for the final and preliminary materialized CHIRPS analysis datasets.

    Subclasses set `template_config`, `region_job_class` and the schedule / latency
    class attributes.
    """

    region_job_class: type[UcsbChcChirpsAnalysisMaterializedRegionJob]

    update_schedule: ClassVar[str]
    validate_schedule: ClassVar[str]
    max_expected_delay: ClassVar[timedelta]
    update_deadline: ClassVar[timedelta] = timedelta(minutes=60)

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule=self.update_schedule,
            pod_active_deadline=self.update_deadline,
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="38G",
            shared_memory="25.5G",
            ephemeral_storage="20G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule=self.validate_schedule,
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )
        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=self.max_expected_delay),
            validation.CheckRecentNans(
                # Random point sampling drops the 71.9% of the grid that is ocean,
                # NaN on every day, leaving land, where a day is either wholly read
                # or missing. 40 points makes an all-ocean sample, which fails the
                # check for want of signal, vanishingly unlikely.
                max_nan_fraction=0.05,
                sampled_points=40,
                append_dim_window=3,
            ),
        )
