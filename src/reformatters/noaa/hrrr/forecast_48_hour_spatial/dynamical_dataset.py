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
    NoaaHrrrForecast48HourSpatialRegionJob,
    NoaaHrrrForecast48HourSpatialSourceFileCoord,
)
from .template_config import NoaaHrrrForecast48HourSpatialTemplateConfig


class NoaaHrrrForecast48HourSpatialDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrForecast48HourSpatialSourceFileCoord]
):
    """NOAA HRRR 48-hour spatial (virtual icechunk) forecast dataset."""

    template_config: NoaaHrrrForecast48HourSpatialTemplateConfig = (
        NoaaHrrrForecast48HourSpatialTemplateConfig()
    )
    region_job_class: type[NoaaHrrrForecast48HourSpatialRegionJob] = (
        NoaaHrrrForecast48HourSpatialRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=(
                icechunk.VirtualChunkContainer(
                    _S3_LOCATION_PREFIX, icechunk.s3_store(region=_S3_BUCKET_REGION)
                ),
            ),
            # Sized per group from measured ~16.4 bytes/ref so full manifests land
            # comfortably under the reader budgets (single-level <= 3 MiB/var, vertical
            # 5-8 MiB) while keeping the total manifest count low; see "Manifest
            # splitting" in docs/virtual_datasets.md. Full-window sizes: single-level
            # 2400 x 49 refs/init ~= 1.8 MiB, pressure 180 x 1911 ~= 5.4 MiB,
            # model 160 x 2450 ~= 6.1 MiB.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 180,
                    r"^/model_level/": 160,
                    None: 2400,
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
            cpu="1.5",
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
