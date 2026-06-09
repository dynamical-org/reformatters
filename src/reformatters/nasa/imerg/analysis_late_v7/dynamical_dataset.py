from collections.abc import Sequence
from datetime import timedelta
from functools import partial

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import (
    CronJob,
    ReformatCronJob,
    ValidationCronJob,
)
from reformatters.nasa.imerg.region_job import NasaImergSourceFileCoord
from reformatters.nasa.imerg.template_config import NasaImergDataVar

from .region_job import NasaImergAnalysisLateV7RegionJob
from .template_config import NasaImergAnalysisLateV7TemplateConfig


class NasaImergAnalysisLateV7Dataset(
    DynamicalDataset[NasaImergDataVar, NasaImergSourceFileCoord]
):
    template_config: NasaImergAnalysisLateV7TemplateConfig = (
        NasaImergAnalysisLateV7TemplateConfig()
    )
    region_job_class: type[NasaImergAnalysisLateV7RegionJob] = (
        NasaImergAnalysisLateV7RegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        # Late Run has ~12-14 hour latency. Each update reprocesses the current 30-day
        # time shard (see template_config chunk comments), so the pod is sized to hold
        # one full-spatial time shard in shared memory.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="0 2,14 * * *",
            pod_active_deadline=timedelta(minutes=90),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="16",
            # Per-worker buffer is ~37GB (one full-spatial 30-day time shard, see
            # template_config); give shared_memory headroom above it and 1.5x for heap.
            memory="63G",
            shared_memory="42G",
            ephemeral_storage="60G",
            # Earthdata credentials required to download source data from GES DISC.
            secret_names=[*self.store_factory.k8s_secret_names(), "nasa-earthdata"],
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="30 3,15 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        # Late Run latency is ~12-14h; allow extra buffer to suppress inactionable alerts.
        max_expected_delay = timedelta(hours=24)
        return (
            partial(
                validation.check_analysis_current_data,
                max_expected_delay=max_expected_delay,
            ),
            partial(
                validation.check_analysis_recent_nans,
                max_expected_delay=max_expected_delay,
                # IMERG has structural NaNs (data gaps poleward of 60 deg and where no
                # precipitation estimate exists). This loose bound should be tuned once
                # real NaN fractions are observed.
                max_nan_fraction=0.95,
                spatial_sampling="quarter",
            ),
        )
