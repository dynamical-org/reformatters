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
            # Sized so refs per commit (arrays touched x refs per split) sits in
            # line with the operationally tuned HRRR virtual datasets: 1.6M here, well
            # under the 10.2-10.7M those three carry, whose end-to-end publish to
            # reader-visible latency is measured in production at p50 2.8s. Coarser is
            # also protective: fewer manifests means a smaller manifest scan, which is
            # superlinear in the deployed icechunk version. Every vertical group needs
            # its own entry -- the catch-all is sized for root arrays, so a group that
            # falls through silently gets a window multiplied by its level count.
            # Re-windowing after a change rewrites every touched array's history in one
            # commit, so treat as frozen.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 512,
                    r"^/height_above_mean_sea_level/": 4096,
                    None: 4096,
                },
                dim="time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # A cycle's f000 and f006 both publish ~init+3h50m, and the six hours the cycle
        # completes are not visible until the last of them lands; the run polls from its
        # fire until they do.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="50 3,9,15,21 * * *",
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
            schedule="35 4,10,16,22 * * *",
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
            # A cycle publishes six hours of analysis ~3h50m after its initialization,
            # the newest of them stamped 5 hours after it, so just before the next cycle
            # lands the newest time is ~5h old. 11 hours also covers one missed cycle.
            validation.CheckCurrentData(max_delay=timedelta(hours=11)),
            # discover_available holds the frontier back to a whole hour, but releases
            # an earlier incomplete hour once a later one is complete, so an interior
            # gap is reachable and this is what finds it.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
