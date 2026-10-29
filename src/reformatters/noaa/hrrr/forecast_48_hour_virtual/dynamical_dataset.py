from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.materialized_update_trigger import MaterializedUpdateTrigger
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
    hrrr_virtual_chunk_containers,
)

from .region_job import NoaaHrrrForecast48HourVirtualRegionJob
from .template_config import NoaaHrrrForecast48HourVirtualTemplateConfig


class NoaaHrrrForecast48HourVirtualDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrForecastVirtualSourceFileCoord]
):
    """NOAA HRRR 48-hour virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: NoaaHrrrForecast48HourVirtualTemplateConfig = (
        NoaaHrrrForecast48HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaHrrrForecast48HourVirtualRegionJob] = (
        NoaaHrrrForecast48HourVirtualRegionJob
    )

    materialized_update_triggers: ClassVar[tuple[MaterializedUpdateTrigger, ...]] = (
        MaterializedUpdateTrigger(
            target_cronjob="noaa-hrrr-forecast-48-hour-update",
            init_hours=(0, 6, 12, 18),
        ),
    )
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(seconds=83)

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=hrrr_virtual_chunk_containers(),
            # Sized for operational commit latency: active-window manifest bytes bound
            # per-commit flush cost. Full-window sizes at ~16.4 bytes/ref: single-level
            # 600 x 49 refs/init ~= 0.55 MiB, pressure 90 x 1911 ~= 2.8 MiB, model
            # 80 x 2450 ~= 3.2 MiB; see "Manifest splitting" in docs/virtual_datasets.md
            # for the cost model.
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
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="30 0,6,12,18 * * *",
            concurrency_policy="Allow",
            pod_active_deadline=timedelta(hours=6, minutes=3, seconds=8),
            active_deadline=timedelta(hours=6, minutes=3, seconds=8),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(minutes=74)),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
