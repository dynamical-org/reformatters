from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.ucsb_chc.chirps.chirps_config_models import UcsbChcChirpsDataVar
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisMaterializedRegionJob,
    UcsbChcChirpsAnalysisSourceFileCoord,
)


class UcsbChcChirpsAnalysisMaterializedDataset(
    DynamicalDataset[UcsbChcChirpsDataVar, UcsbChcChirpsAnalysisSourceFileCoord]
):
    """Shared base for the final and preliminary materialized CHIRPS analysis datasets.

    Subclasses set `template_config`, `region_job_class` and the schedule / latency
    class attributes.
    """

    region_job_class: type[UcsbChcChirpsAnalysisMaterializedRegionJob]

    update_schedule: ClassVar[str]
    validate_schedule: ClassVar[str]
    update_deadline: ClassVar[timedelta]
    max_expected_delay: ClassVar[timedelta]

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
                # CHIRPS estimates land only: 71.9% of the grid is NaN, and the worst
                # quarter (southern hemisphere west of the prime meridian) is 87.5%.
                # The land/water mask is identical on every day, so 0.89 leaves room
                # for the sampled quarter and still fails a day a source file is
                # missing from.
                max_nan_fraction=0.89,
                spatial_sampling="quarter",
                append_dim_window=3,
            ),
        )
