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

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=gefs_virtual_chunk_containers(),
            # Eight days of daily inits at the root, fewer for the vertical groups.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 2,
                    r"^/model_level/": 4,
                    r"^/height_above_mean_sea_level/": 4,
                    None: 8,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # The dataset id plus "-validate" exceeds the 52 character cron job name limit.
        cron_job_name_prefix = self.dataset_id.replace("-0-5-degree", "-0-5")
        # f000-f384 publishes ~init+3h46m through ~init+6h43m; f390-f840 publishes in
        # bursts until ~init+28h05m. Fire just before the first burst; the 6h deadline
        # covers it, and the extension is ingested as soon as the next day's fire finds it.
        operational_update_cron_job = ReformatCronJob(
            name=f"{cron_job_name_prefix}-update",
            schedule="45 3 * * *",
            pod_active_deadline=timedelta(hours=6),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            # A fire opens by ingesting the whole extension of the previous cycle,
            # the largest single batch of refs it holds.
            memory="15G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=1,
            parallelism=1,
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{cron_job_name_prefix}-validate",
            # The update's fire plus its pod_active_deadline
            schedule="55 9 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=9, minutes=50)),
            # 00z has published only its leads through 384h, 105 of 181, when validation fires.
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.57, 1.0)
            ),
            validation.CheckVirtualDecodeHealth(),
        )
