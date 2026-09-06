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
            schedule="25 4,10,16,22 * * *",
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
            # A 00, 06, 12 or 18 hour waits on its own cycle, so a position lands
            # ~4h15m after its timestamp at the latest. Keep this under the 6h cycle
            # spacing or a wholly missed cycle is not yet due at the next validation run.
            validation.CheckCurrentData(max_delay=timedelta(hours=4, minutes=20)),
            # Every ingested position is whole, so no leading fraction tier is needed.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
