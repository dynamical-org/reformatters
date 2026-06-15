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
    _S3_BUCKET_REGION,
    _S3_LOCATION_PREFIX,
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
                    _S3_LOCATION_PREFIX, icechunk.s3_store(region=_S3_BUCKET_REGION)
                ),
            ),
            # One week of inits per manifest split: every commit rewrites each
            # touched array's active split, so the split size caps commit cost
            # (~1.3 MB per array at week end). The -dev operational test
            # measures this tradeoff; see "Manifest splitting" in the plan.
            manifest_split=manifest_append_dim_split(split_size=7 * 4, dim="init_time"),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Single-writer; one fire per init, 3 minutes before the earliest observed
        # publication start (init+3:46), polling through the publication window.
        # The pod exits when the init is fully ingested (observed init+5:30-5:39);
        # the deadline (init+5:53) bounds waiting on a file that never publishes.
        # See "Publication timing measurements" in the virtual datasets plan.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="43 3,9,15,21 * * *",
            # Must stay well under the 6h between fires so fires never overlap.
            pod_active_deadline=timedelta(hours=2, minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.7",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # Validate after each update: update fire (43 3,9,15,21) + its 2h10m deadline.
            schedule="53 5,11,17,23 * * *",
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
