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
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.virtual_region_job import hrrr_virtual_chunk_containers

from .region_job import (
    NoaaHrrrAnalysisVirtualRegionJob,
    NoaaHrrrAnalysisVirtualSourceFileCoord,
)
from .template_config import NoaaHrrrAnalysisVirtualTemplateConfig


class NoaaHrrrAnalysisVirtualDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrAnalysisVirtualSourceFileCoord]
):
    """NOAA HRRR virtual (spatially-chunked, map-optimized icechunk) analysis dataset."""

    template_config: NoaaHrrrAnalysisVirtualTemplateConfig = (
        NoaaHrrrAnalysisVirtualTemplateConfig()
    )
    region_job_class: type[NoaaHrrrAnalysisVirtualRegionJob] = (
        NoaaHrrrAnalysisVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=hrrr_virtual_chunk_containers(),
            # At ~16.4 bytes/ref, each manifest holds about 0.47 MiB for single-level,
            # 2.7 MiB for pressure-level, or 3.1 MiB for model-level arrays.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 4500,
                    r"^/model_level/": 4000,
                    None: 30000,
                },
                dim="time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Race each hour's cycle: f00/wrfprs/wrfnat publish ~init+51m and the prior
        # cycle's f01 is already out (~init+53m), so a :50 fire commits the hour's
        # files within minutes; a late cycle rolls to the next fire.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.cron_job_name_prefix}-update",
            schedule="50 * * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.cron_job_name_prefix}-validate",
            # The update's fire plus its pod_active_deadline, so the run being
            # validated has always stopped writing.
            schedule="20 * * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            # An hour is ingested when its own f00 files publish, ~1h after its
            # timestamp. Two hours leaves room for one cycle to roll to the next fire.
            validation.CheckCurrentData(max_delay=timedelta(hours=2)),
            # discover_available extends time only to an hour holding every file it
            # needs, so every ingested position is whole.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(
                allow_all_nan_vars=(
                    "echo_top",
                    "percent_frozen_precipitation_surface",
                )
            ),
        )
