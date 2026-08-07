from collections.abc import Sequence
from datetime import timedelta
from functools import partial

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
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
        # Run once per 6h cycle just before the files publish: all 61 leads land in
        # a ~2 minute burst ~init+5h25m to init+6h03m (S3 LastModified; p99
        # ~init+6h10m). Fire at init+5h20m and poll; the 1h deadline covers p99,
        # bounds waiting on a cycle that never publishes, and keeps fires from
        # overlapping.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="20 5,11,17,23 * * *",
            pod_active_deadline=timedelta(hours=1),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.7",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # After each update (init+5h20m) + its 1h deadline.
            schedule="20 0,6,12,18 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        # Validation fires at init+6h20m, just after the update deadline; files
        # publish by ~init+6h10m (p99), so 7h leaves ~40m of cron/pod start slack.
        return (
            partial(
                validation.check_forecast_current_data,
                max_latest_init_time_age=timedelta(hours=7),
            ),
            # All 61 leads land in a ~2 minute burst, so an ingested init is a whole one.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
