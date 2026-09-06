from collections.abc import Sequence
from datetime import timedelta

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.noaa.gfs.virtual_region_job import gfs_virtual_chunk_containers
from reformatters.noaa.models import NoaaDataVar

from .region_job import (
    NoaaGfsAnalysisVirtualRegionJob,
    NoaaGfsAnalysisVirtualSourceFileCoord,
)
from .template_config import NoaaGfsAnalysisVirtualTemplateConfig


class NoaaGfsAnalysisVirtualDataset(
    DynamicalDataset[NoaaDataVar, NoaaGfsAnalysisVirtualSourceFileCoord]
):
    """NOAA GFS virtual (spatially-chunked, map-optimized icechunk) analysis dataset."""

    template_config: NoaaGfsAnalysisVirtualTemplateConfig = (
        NoaaGfsAnalysisVirtualTemplateConfig()
    )
    region_job_class: type[NoaaGfsAnalysisVirtualRegionJob] = (
        NoaaGfsAnalysisVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=gfs_virtual_chunk_containers(),
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 3_000,
                    r"^/height_above_mean_sea_level/": 20_000,
                    None: 30_000,
                },
                dim="time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # This analysis uses leads 0-6, which all publish ~init+3h32m to ~init+3h53m.
        # Fire a few minutes before the earliest and poll until the last one lands.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="29 3,9,15,21 * * *",
            pod_active_deadline=timedelta(minutes=45),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # The update's fire plus its pod_active_deadline, so the run being
            # validated has always stopped writing.
            schedule="14 4,10,16,22 * * *",
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
            # A 00, 06, 12 or 18 hour waits on its own cycle, so a position lands
            # ~4h15m after its timestamp at the latest. Keep this under the 6h cycle
            # spacing or a wholly missed cycle is not yet due at the next validation run.
            validation.CheckCurrentData(max_delay=timedelta(hours=5, minutes=30)),
            # discover_available holds the frontier back to a whole hour, but releases
            # an earlier incomplete hour once a later one is complete, so an interior
            # gap is reachable and this is what finds it.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
