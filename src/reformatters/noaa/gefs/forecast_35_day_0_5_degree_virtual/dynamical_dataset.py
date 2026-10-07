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
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsVirtualDataVar
from reformatters.noaa.gefs.virtual_region_job import (
    NoaaGefsForecastVirtualSourceFileCoord,
    gefs_virtual_chunk_containers,
)

from .region_job import NoaaGefsForecast35Day05DegreeVirtualRegionJob
from .template_config import NoaaGefsForecast35Day05DegreeVirtualTemplateConfig


class NoaaGefsForecast35Day05DegreeVirtualDataset(
    DynamicalDataset[NoaaGefsVirtualDataVar, NoaaGefsForecastVirtualSourceFileCoord]
):
    """NOAA GEFS 35 day 0.5 degree virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: NoaaGefsForecast35Day05DegreeVirtualTemplateConfig = (
        NoaaGefsForecast35Day05DegreeVirtualTemplateConfig()
    )
    region_job_class: type[NoaaGefsForecast35Day05DegreeVirtualRegionJob] = (
        NoaaGefsForecast35Day05DegreeVirtualRegionJob
    )
    # The newest extension remains partial throughout the polling window.
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=7)

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=gefs_virtual_chunk_containers(),
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 2,
                    r"^/model_level/": 4,
                    r"^/height_above_mean_sea_level/": 4,
                    None: 4,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        cron_job_name_prefix = self.dataset_id.replace("-0-5-degree", "-0-5")
        # f000-f384 publishes ~init+3h46m to +6h43m. Extension member waves start
        # near init+6h30m, +12h30m and +18h30m; overnight dumps finish around
        # init+27h06m to +28h11m. Poll before the previous cycle finishes and through
        # the newest morning extension wave, which can end around init+10h06m.
        operational_update_cron_job = ReformatCronJob(
            name=f"{cron_job_name_prefix}-update",
            schedule="15 2 * * *",
            pod_active_deadline=timedelta(hours=8, minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            # A fire opens by catching up most of the previous cycle's extension,
            # the largest single batch of refs it holds.
            memory="15G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=1,
            parallelism=1,
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=3, minutes=50)),
            # The newest init can be partial while its long-lead extension publishes.
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.57, 1.0)
            ),
            # Four levels so the sample reaches 600 hPa, an icing level; three would
            # miss every level icing is published at.
            validation.CheckVirtualDecodeHealth(sampled_levels=4),
        )
