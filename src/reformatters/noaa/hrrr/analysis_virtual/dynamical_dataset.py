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
            # The analysis carries one lead per time position where the 48-hour
            # product carries 49 per init, so its splits scale up ~49x to keep full
            # manifest byte sizes comparable: at ~16.4 bytes/ref, single-level
            # 30000 x 1 ref/time ~= 0.47 MiB, pressure 4500 x 39 ~= 2.7 MiB, model
            # 4000 x 50 ~= 3.1 MiB; see "Manifest splitting" in
            # docs/virtual_datasets.md for the cost model.
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
            name=f"{self.dataset_id}-update",
            schedule="50 * * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
            # Suspended until the initial backfill completes.
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # The update's fire plus its pod_active_deadline, so the run being
            # validated has always stopped writing.
            schedule="20 * * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
            # Suspended until the initial backfill completes.
            suspend=True,
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        hour_0_var_paths = tuple(
            var.path
            for var in self.template_config.data_vars
            if var.has_hour_0_values()
        )
        return (
            partial(
                validation.check_analysis_current_data,
                max_expected_delay=timedelta(hours=2),
            ),
            validation.CheckVirtualManifestCompleteness(exclude_vars=hour_0_var_paths),
            # A cycle running past the update's poll deadline leaves the newest hour
            # without its own f00 files until the next fire.
            validation.CheckVirtualManifestCompleteness(
                include_vars=hour_0_var_paths,
                min_present_fraction=(0.0, 1.0),
            ),
            validation.CheckVirtualDecodeHealth(),
        )
