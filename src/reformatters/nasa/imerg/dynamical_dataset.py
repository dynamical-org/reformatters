from collections.abc import Sequence
from datetime import timedelta
from functools import partial
from typing import ClassVar

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.nasa.imerg.imerg_config_models import NasaImergDataVar
from reformatters.nasa.imerg.region_job import (
    NasaImergAnalysisMaterializedRegionJob,
    NasaImergAnalysisSourceFileCoord,
)


class NasaImergAnalysisMaterializedDataset(
    DynamicalDataset[NasaImergDataVar, NasaImergAnalysisSourceFileCoord]
):
    """Shared base for the Early and Late materialized IMERG analysis datasets.

    Subclasses set `template_config`, `region_job_class` and the schedule / latency
    class attributes.
    """

    region_job_class: type[NasaImergAnalysisMaterializedRegionJob]

    update_schedule: ClassVar[str]
    validate_schedule: ClassVar[str]
    max_expected_delay: ClassVar[timedelta]

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule=self.update_schedule,
            pod_active_deadline=timedelta(minutes=60),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="7",
            memory="62G",
            shared_memory="38G",
            ephemeral_storage="20G",
            # PPS NRT (jsimpson) for low-latency granules, GES DISC (Earthdata) fallback.
            secret_names=[
                *self.store_factory.k8s_secret_names(),
                "nasa-pps",
                "nasa-earthdata",
            ],
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule=self.validate_schedule,
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="4G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        return (
            partial(
                validation.check_analysis_current_data,
                max_expected_delay=self.max_expected_delay,
            ),
            partial(
                validation.check_analysis_recent_nans,
                max_expected_delay=self.max_expected_delay,
                # IMERG is globally complete (precip is 0, not NaN, where it is dry);
                # only sparse polar gaps are NaN. On-disk granules measure <=0.8% NaN
                # globally (deep archive) and <=0.15% recent; "quarter" sampling can
                # concentrate the polar gaps into one hemisphere, roughly doubling that.
                # 0.05 leaves margin for seasonal polar variation the sample can't see.
                max_nan_fraction=0.05,
                spatial_sampling="quarter",
            ),
        )
