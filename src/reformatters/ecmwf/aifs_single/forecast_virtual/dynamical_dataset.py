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

from .region_job import (
    EcmwfAifsSingleForecastVirtualRegionJob,
    EcmwfAifsSingleForecastVirtualSourceFileCoord,
    aifs_single_virtual_chunk_containers,
)
from .template_config import (
    EcmwfAifsSingleForecastVirtualTemplateConfig,
    EcmwfAifsSingleVirtualDataVar,
)


class EcmwfAifsSingleForecastVirtualDataset(
    DynamicalDataset[
        EcmwfAifsSingleVirtualDataVar, EcmwfAifsSingleForecastVirtualSourceFileCoord
    ]
):
    """ECMWF AIFS Single virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: EcmwfAifsSingleForecastVirtualTemplateConfig = (
        EcmwfAifsSingleForecastVirtualTemplateConfig()
    )
    region_job_class: type[EcmwfAifsSingleForecastVirtualRegionJob] = (
        EcmwfAifsSingleForecastVirtualRegionJob
    )

    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(seconds=44)

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=aifs_single_virtual_chunk_containers(),
            # Sized for operational commit latency: active-window manifest bytes bound
            # per-commit flush cost. Full-window sizes at ~16.4 bytes/ref: single-level
            # 600 x 61 refs/init ~= 0.6 MiB, pressure 200 x 854 (61 leads x 14 levels)
            # ~= 2.7 MiB; see "Manifest splitting" in docs/virtual_datasets.md for the
            # cost model.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 200,
                    None: 600,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="49 4,10,16,22 * * *",
            concurrency_policy="Allow",
            pod_active_deadline=timedelta(hours=6, minutes=2),
            active_deadline=timedelta(hours=6, minutes=2),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.7",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=6, minutes=6)),
            # Polling waits for all 61 leads before validating a complete run.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
