from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from pydantic import Field, NonNegativeInt, PositiveInt

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.noaa.rrfs.models import NoaaRrfsDataVar
from reformatters.noaa.rrfs.region_job import (
    NoaaRrfsSourceFileCoord,
    rrfs_virtual_chunk_containers,
)
from reformatters.noaa.rrfs.template_config import NoaaRrfsForecastTemplateConfig


class NoaaRrfsDataset(DynamicalDataset[NoaaRrfsDataVar, NoaaRrfsSourceFileCoord]):
    template_config: NoaaRrfsForecastTemplateConfig
    update_offset_minutes: NonNegativeInt
    update_deadline_minutes: PositiveInt
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

        hours = ",".join(
            str(h)
            for h in sorted(
                {
                    (h + self.update_offset_minutes // 60) % 24
                    for h in range(0, 24, cadence)
                }
            )
        )

        return (
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule=f"{self.update_offset_minutes % 60} {hours} * * *",
                pod_active_deadline=timedelta(minutes=self.update_deadline_minutes),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1.5",
                memory="3.7G",
                secret_names=self.store_factory.k8s_secret_names(),
            ),
        )

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(
                max_delay=timedelta(
                    minutes=self.update_offset_minutes + self.update_deadline_minutes
                )
                - self.virtual_poll_deadline_grace
            ),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(max_workers=2),
        )
