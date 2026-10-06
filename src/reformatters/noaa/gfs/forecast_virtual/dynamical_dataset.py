from collections.abc import Sequence
from datetime import timedelta

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.noaa.gfs.virtual_region_job import gfs_virtual_chunk_containers
from reformatters.noaa.models import NoaaDataVar

from .region_job import (
    NoaaGfsForecastVirtualRegionJob,
    NoaaGfsForecastVirtualSourceFileCoord,
)
from .template_config import NoaaGfsForecastVirtualTemplateConfig


class NoaaGfsForecastVirtualDataset(
    DynamicalDataset[NoaaDataVar, NoaaGfsForecastVirtualSourceFileCoord]
):
    """NOAA GFS virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: NoaaGfsForecastVirtualTemplateConfig = (
        NoaaGfsForecastVirtualTemplateConfig()
    )
    region_job_class: type[NoaaGfsForecastVirtualRegionJob] = (
        NoaaGfsForecastVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=gfs_virtual_chunk_containers(),
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 4,
                    r"^/height_above_mean_sea_level/": 8,
                    None: 16,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # A full forecast publishes ~init+3h32m to ~init+5h34m.
        # Fire a few minutes before the earliest and poll until the last one lands. The
        # deadline bounds waiting on a file that never publishes.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="29 3,9,15,21 * * *",
            pod_active_deadline=timedelta(hours=2, minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=3, minutes=33)),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
