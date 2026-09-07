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

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=gefs_virtual_chunk_containers(),
            # Two days of 6 hourly inits at the root, about one for the vertical groups.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 4,
                    r"^/model_level/": 4,
                    r"^/height_above_mean_sea_level/": 5,
                    None: 8,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # The dataset id plus "-validate" exceeds the 52 character cron job name limit.
        cron_job_name_prefix = self.dataset_id.replace("-0-5-degree", "-0-5")
        # The whole run publishes in one burst: the first file lands ~init+3h46m and the
        # last member's f384 ~init+7h11m. Fire just before the burst starts and poll
        # through it; the deadline clears the observed end by over half an hour and
        # still ends well before the next cycle's fire.
        operational_update_cron_job = ReformatCronJob(
            name=f"{cron_job_name_prefix}-update",
            schedule="45 3,9,15,21 * * *",
            pod_active_deadline=timedelta(hours=4),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=1,
            parallelism=1,
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{cron_job_name_prefix}-validate",
            # The update's fire plus its pod_active_deadline, so the run being
            # validated has always stopped writing.
            schedule="55 7,13,19,1 * * *",
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
            # A cycle that published nothing is caught here rather than by the
            # completeness check below, which skips append dim positions the store
            # does not reach.
            validation.CheckCurrentData(max_delay=timedelta(hours=7, minutes=50)),
            # The whole run publishes before validation fires, so every init the store
            # reached must be whole.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
