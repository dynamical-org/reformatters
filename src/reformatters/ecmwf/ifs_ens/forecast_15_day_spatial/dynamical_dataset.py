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
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar

from .region_job import (
    _S3_BUCKET_REGION,
    _S3_LOCATION_PREFIX,
    EcmwfIfsEnsForecast15DaySpatialRegionJob,
    IfsEnsForecast15DaySpatialSourceFileCoord,
)
from .template_config import EcmwfIfsEnsForecast15DaySpatialTemplateConfig


class EcmwfIfsEnsForecast15DaySpatialDataset(
    DynamicalDataset[EcmwfDataVar, IfsEnsForecast15DaySpatialSourceFileCoord]
):
    """ECMWF IFS ENS 15-day spatial (virtual icechunk) forecast dataset."""

    template_config: EcmwfIfsEnsForecast15DaySpatialTemplateConfig = (
        EcmwfIfsEnsForecast15DaySpatialTemplateConfig()
    )
    region_job_class: type[EcmwfIfsEnsForecast15DaySpatialRegionJob] = (
        EcmwfIfsEnsForecast15DaySpatialRegionJob
    )
    # default_factory because icechunk's container objects can't be deep-copied
    # as a plain pydantic default would be.
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=(
                icechunk.VirtualChunkContainer(
                    _S3_LOCATION_PREFIX, icechunk.s3_store(region=_S3_BUCKET_REGION)
                ),
            ),
            # One week of (daily) inits per manifest split: every commit rewrites
            # each touched array's active split, so the split size caps commit cost.
            manifest_split=manifest_append_dim_split(split_size=7, dim="init_time"),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Single-writer; one fire per day's 00z init. ECMWF uploads the first
        # open-data file at ~07:40 UTC and the last by ~07:45 UTC; the pod streams
        # files as they appear and exits once the init is fully ingested, with the
        # deadline bounding waiting on a file that never publishes.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="45 7 * * *",
            pod_active_deadline=timedelta(hours=1, minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.7",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # Validate after the update: update fire (45 7) + its 1h30m deadline.
            schedule="15 9 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        # Minimal for now: the NaN validators decode one full GRIB message per
        # chunk touched, which is unbounded on a virtual store. Manifest-aware
        # validators are planned (virtual datasets plan, PR 8).
        # 24h cycle, 00z published ~07:40 + ingest -> healthy latest-init age ~9h at
        # validate; 12h alerts on a missed cycle without firing on publish slippage.
        return (
            partial(
                validation.check_forecast_current_data,
                max_latest_init_time_age=timedelta(hours=12),
            ),
        )
