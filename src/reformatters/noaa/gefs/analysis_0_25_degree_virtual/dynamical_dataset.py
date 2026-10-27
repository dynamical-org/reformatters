from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

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

    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(seconds=38)

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
            schedule="44 3,9,15,21 * * *",
            concurrency_policy="Allow",
            pod_active_deadline=timedelta(hours=6, minutes=2, seconds=27),
            active_deadline=timedelta(hours=6, minutes=2, seconds=27),
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
            validation.CheckCurrentData(max_delay=timedelta(minutes=249)),
            # Every ingested position is whole, so no leading fraction tier is needed.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
