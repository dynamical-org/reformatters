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

from .region_job import NoaaGefsForecast16Day05DegreeVirtualRegionJob
from .template_config import NoaaGefsForecast16Day05DegreeVirtualTemplateConfig


class NoaaGefsForecast16Day05DegreeVirtualDataset(
    DynamicalDataset[NoaaGefsVirtualDataVar, NoaaGefsForecastVirtualSourceFileCoord]
):
    """NOAA GEFS 16 day 0.5 degree virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: NoaaGefsForecast16Day05DegreeVirtualTemplateConfig = (
        NoaaGefsForecast16Day05DegreeVirtualTemplateConfig()
    )
    region_job_class: type[NoaaGefsForecast16Day05DegreeVirtualRegionJob] = (
        NoaaGefsForecast16Day05DegreeVirtualRegionJob
    )

    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=6, seconds=33)

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=gefs_virtual_chunk_containers(),
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 2,
                    r"^/model_level/": 2,
                    r"^/height_above_mean_sea_level/": 2,
                    None: 4,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        cron_job_name_prefix = self.dataset_id.replace("-0-5-degree", "-0-5")
        # A run publishes ~init+3h46m through ~init+7h11m.
        # Poll across the next fire while files remain pending.
        operational_update_cron_job = ReformatCronJob(
            name=f"{cron_job_name_prefix}-update",
            schedule="29 3,9,15,21 * * *",
            concurrency_policy="Allow",
            pod_active_deadline=timedelta(hours=6, minutes=8, seconds=22),
            active_deadline=timedelta(hours=6, minutes=8, seconds=22),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=1,
            parallelism=1,
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=4, minutes=8)),
            validation.CheckVirtualManifestCompleteness(),
            # Four levels so the sample reaches 600 hPa, an icing level; three would
            # miss every level icing is published at.
            validation.CheckVirtualDecodeHealth(sampled_levels=4),
        )
