from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.noaa.rrfs.models import NoaaRrfsDataVar
from reformatters.noaa.rrfs.region_job import (
    NoaaRrfsSourceFileCoord,
    rrfs_virtual_chunk_containers,
)
from reformatters.noaa.rrfs.template_config import NoaaRrfsForecastTemplateConfig


class NoaaRrfsDataset(DynamicalDataset[NoaaRrfsDataVar, NoaaRrfsSourceFileCoord]):
    template_config: NoaaRrfsForecastTemplateConfig
    _operational_timing: ClassVar[tuple[int, int]]
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=5)
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=rrfs_virtual_chunk_containers(),
            manifest_split=manifest_append_dim_split(
                # At ~16 B/ref, pressure manifests are ~5.3 MiB deterministic / ~7.0 MiB ensemble;
                # 300-init AMSL/depth manifests are ~3.9/3.5 MiB, with fewer root manifests.
                split_size={
                    r"^/pressure_level/": 90,
                    r"^/height_above_mean_sea_level/": 300,
                    r"^/depth_below_ground/": 300,
                    None: 300,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        cadence = int(self.template_config.append_dim_frequency.total_seconds() / 3600)
        offset_minutes, deadline_minutes = self._operational_timing

        def schedule(offset: int) -> str:
            hours = ",".join(
                str(h)
                for h in sorted(
                    {(h + offset // 60) % 24 for h in range(0, 24, cadence)}
                )
            )
            return f"{offset % 60} {hours} * * *"

        return (
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule=schedule(offset_minutes),
                pod_active_deadline=timedelta(minutes=deadline_minutes),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="2",
                memory="3.7G",
                secret_names=self.store_factory.k8s_secret_names(),
                suspend=True,
            ),
            ValidationCronJob(
                name=f"{self.dataset_id}-validate",
                schedule=schedule(offset_minutes + deadline_minutes + 5),
                pod_active_deadline=timedelta(minutes=30),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="3.7G",
                secret_names=self.store_factory.k8s_secret_names(),
                suspend=True,
            ),
        )

    def validators(self) -> Sequence[validation.Validator]:
        offset_minutes, deadline_minutes = self._operational_timing
        return (
            validation.CheckCurrentData(
                max_delay=timedelta(minutes=offset_minutes + deadline_minutes + 5)
            ),
            # Hourly windows include an unpublished next init; the newest ingested init is normally the previous one.
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.05, 1.0)
                if self.template_config.sub_hourly
                else (1.0,),
            ),
            validation.CheckVirtualDecodeHealth(max_workers=2),
        )
