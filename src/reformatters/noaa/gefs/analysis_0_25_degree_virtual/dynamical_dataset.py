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
            # Four years of 3-hourly steps. Refs per commit = arrays x refs per active
            # split, order 10^7; see "Manifest splitting" in docs/virtual_datasets.md.
            manifest_split=manifest_append_dim_split(
                split_size=4 * 365 * 8, dim="time"
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # f006, the longest lead an analysis step uses, publishes ~init+3h48m.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="51 3,9,15,21 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.7",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=1,
            parallelism=1,
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # The update's fire plus its pod_active_deadline, so the run being
            # validated has always stopped writing.
            schedule="31 4,10,16,22 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            # A time is ingested when its own shortest-lead file publishes, hours
            # after the cycle that covers it starts.
            validation.CheckCurrentData(max_delay=timedelta(hours=13)),
            # Every ingested position is whole, so no leading fraction tier is needed.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
