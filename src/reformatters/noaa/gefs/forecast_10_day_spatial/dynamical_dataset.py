from collections.abc import Sequence
from datetime import timedelta

import icechunk
from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

from .region_job import (
    GefsForecast10DaySpatialRegionJob,
    GefsForecast10DaySpatialSourceFileCoord,
)
from .template_config import GefsForecast10DaySpatialTemplateConfig


class GefsForecast10DaySpatialDataset(
    DynamicalDataset[GEFSDataVar, GefsForecast10DaySpatialSourceFileCoord]
):
    """GEFS 10-day spatial (virtual icechunk) forecast dataset."""

    template_config: GefsForecast10DaySpatialTemplateConfig = (
        GefsForecast10DaySpatialTemplateConfig()
    )
    region_job_class: type[GefsForecast10DaySpatialRegionJob] = (
        GefsForecast10DaySpatialRegionJob
    )
    # default_factory because icechunk's container objects can't be deep-copied
    # as a plain pydantic default would be.
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=(
                icechunk.VirtualChunkContainer(
                    "s3://noaa-gefs-pds/", icechunk.s3_store(region="us-east-1")
                ),
            ),
            # One year of inits per manifest split.
            manifest_split=manifest_append_dim_split(
                split_size=365 * 4, dim="init_time"
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Single-writer operational updates; each fire ingests whatever new files
        # are available, so an hourly cadence ingests each init incrementally as
        # it publishes (~3-4h after init time).
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="10 * * * *",
            pod_active_deadline=timedelta(minutes=50),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="2",
            memory="8G",
            shared_memory="1G",
            ephemeral_storage="10G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="40 7 * * *",  # after the 00z init finishes publishing
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
        # (lead, member) chunk touched, which is unbounded on a virtual store.
        # Manifest-aware validators are planned (virtual datasets plan, PR 7).
        return (validation.check_forecast_current_data,)
