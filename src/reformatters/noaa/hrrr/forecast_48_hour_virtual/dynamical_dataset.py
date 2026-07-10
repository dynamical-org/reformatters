from collections.abc import Sequence
from datetime import timedelta
from functools import partial

import icechunk
from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar

from .region_job import (
    _S3_BUCKET_REGION,
    _S3_LOCATION_PREFIX,
    NoaaHrrrForecast48HourVirtualRegionJob,
    NoaaHrrrForecast48HourVirtualSourceFileCoord,
)
from .template_config import NoaaHrrrForecast48HourVirtualTemplateConfig


class NoaaHrrrForecast48HourVirtualDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrForecast48HourVirtualSourceFileCoord]
):
    """NOAA HRRR 48-hour virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: NoaaHrrrForecast48HourVirtualTemplateConfig = (
        NoaaHrrrForecast48HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaHrrrForecast48HourVirtualRegionJob] = (
        NoaaHrrrForecast48HourVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=(
                icechunk.VirtualChunkContainer(
                    _S3_LOCATION_PREFIX, icechunk.s3_store(region=_S3_BUCKET_REGION)
                ),
            ),
            # Sized for operational commit latency: each commit read-modify-writes the
            # active window's manifest for every changed array (~0.1s CPU/MB, measured),
            # so window bytes bound per-commit flush cost. Full-window sizes at ~16.4
            # bytes/ref: single-level 600 x 49 refs/init ~= 0.55 MiB, pressure 90 x 1911
            # ~= 2.8 MiB, model 80 x 2450 ~= 3.2 MiB — balancing commit latency
            # (sfc/prs/nat ~2.6/2.1/4.5s measured at c=16, 4 cpu, 10ms simulated S3)
            # against manifest count (M ~7.6k) and whole-archive read amplification;
            # see "Manifest splitting" in docs/virtual_datasets.md. On first commit per
            # array after a split-size change, icechunk re-windows that array's existing
            # manifests (one-time, ~1 min measured).
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 90,
                    r"^/model_level/": 80,
                    None: 600,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Run once per 6h cycle just before the first lead times publish
        # (~init+50m), polling through f48 (~init+2h on S3). The pod exits when the
        # window is fully ingested; the deadline bounds waiting on a file that never
        # publishes and stays well under the 6h gap so fires never overlap.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="50 0,6,12,18 * * *",
            pod_active_deadline=timedelta(hours=1, minutes=40),
            image=image_tag,
            dataset_id=self.dataset_id,
            # Commit flush parallelizes manifest rebuilds across arrays
            # (ICECHUNK_COMMIT_MAX_CONCURRENT_NODES); cores bound the CPU side.
            cpu="4",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # After each update (:50) + its 1h40m deadline.
            schedule="40 2,8,14,20 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        # 6h cycle + ~2h publication = ~8h before the latest init is current.
        return (
            partial(
                validation.check_forecast_current_data,
                max_latest_init_time_age=timedelta(hours=8),
            ),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
