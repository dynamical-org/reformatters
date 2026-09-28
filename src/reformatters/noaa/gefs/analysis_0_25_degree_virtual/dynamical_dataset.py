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
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsVirtualDataVar
from reformatters.noaa.gefs.virtual_region_job import gefs_virtual_chunk_containers

from .region_job import (
    NoaaGefsAnalysis025DegreeVirtualRegionJob,
    NoaaGefsAnalysis025DegreeVirtualSourceFileCoord,
)
from .template_config import NoaaGefsAnalysis025DegreeVirtualTemplateConfig


class NoaaGefsAnalysis025DegreeVirtualDataset(
    DynamicalDataset[
        NoaaGefsVirtualDataVar, NoaaGefsAnalysis025DegreeVirtualSourceFileCoord
    ]
):
    """NOAA GEFS virtual (spatially-chunked, map-optimized icechunk) analysis dataset."""

    template_config: NoaaGefsAnalysis025DegreeVirtualTemplateConfig = (
        NoaaGefsAnalysis025DegreeVirtualTemplateConfig()
    )
    region_job_class: type[NoaaGefsAnalysis025DegreeVirtualRegionJob] = (
        NoaaGefsAnalysis025DegreeVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=gefs_virtual_chunk_containers(),
            # Four years of 3-hourly steps.
            manifest_split=manifest_append_dim_split(
                split_size=4 * 365 * 8, dim="time"
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # This analysis uses leads 0-6, which all publish by ~init+3h48m.
        # Fire a few minutes before that and poll until they land.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="45 3,9,15,21 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.7",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=1,
            parallelism=1,
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            # Each update polls the 00, 06, 12 or 18 cycle until its files land or
            # the deadline expires; a wholly missed cycle is due after that update.
            validation.CheckCurrentData(max_delay=timedelta(hours=3, minutes=40)),
            # Every ingested position is whole, so no leading fraction tier is needed.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
