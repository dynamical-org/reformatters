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

from .region_job import NasaImergAnalysisEarlyV7RegionJob
from .template_config import NasaImergAnalysisEarlyV7TemplateConfig


class NasaImergAnalysisEarlyV7Dataset(
    DynamicalDataset[NasaImergDataVar, NasaImergSourceFileCoord]
):
    template_config: NasaImergAnalysisEarlyV7TemplateConfig = (
        NasaImergAnalysisEarlyV7TemplateConfig()
    )
    region_job_class: type[NasaImergAnalysisEarlyV7RegionJob] = (
        NasaImergAnalysisEarlyV7RegionJob
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        # Early Run has ~4 hour latency. Each update reprocesses the current 30-day
        # time shard (see template_config chunk comments), so the pod is sized to hold
        # one full-spatial time shard in shared memory.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="0 0,6,12,18 * * *",
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
            schedule="30 1,7,13,19 * * *",
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
        # Early Run latency is ~4h; allow extra buffer to suppress inactionable alerts.
        max_expected_delay = timedelta(hours=8)
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
