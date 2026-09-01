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
            # An init contributes one ref per lead time to a root array and up to one
            # per lead time and level to a pressure-level one, so both terms of the
            # commit cost (an O(total manifests squared) scan plus a rewrite linear in
            # arrays touched x active manifest bytes) bind well before the reader budget
            # does. These sizes minimize their sum over a fifteen year archive: 0.24 MiB
            # per full root manifest and 1.8 MiB per pressure-level one, both well
            # inside the reader budget. See "Manifest splitting" in
            # docs/virtual_datasets.md; re-windowing after a change is a whole-archive
            # rewrite, so treat these as frozen.
            manifest_split=manifest_append_dim_split(
                split_size={r"^/pressure_level/": 16, None: 128},
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Fire just before a cycle's f000 publishes (~init+3h33m) and poll through f384
        # (~init+5h19m). The pod exits when the window is fully ingested; the deadline
        # bounds waiting on a file that never publishes and keeps fires from overlapping.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="30 3,9,15,21 * * *",
            pod_active_deadline=timedelta(hours=2, minutes=15),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # Ten minutes past the update's fire plus its pod_active_deadline, so the
            # run being validated has always stopped writing.
            schedule="55 5,11,17,23 * * *",
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
            # An init is ingested ~5h20m after it and validation fires at init+5h55m,
            # so 11 hours is the tightest deadline that still absorbs one missed cycle.
            validation.CheckCurrentData(max_delay=timedelta(hours=11)),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
