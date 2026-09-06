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
            # Four days of daily inits at the root, two for the vertical groups, whose
            # arrays hold a ref per level. Refs per commit = arrays x refs per active
            # split; see "Manifest splitting" in docs/virtual_datasets.md.
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
        # The dataset id plus "-validate" exceeds the 52 character cron job name limit,
        # so the resolution takes its source file spelling.
        cron_job_name_prefix = self.dataset_id.replace("-0-5-degree", "-0p5")
        # A cycle publishes in two stages: the lead times through 384 hours land between
        # ~init+3h46m and ~init+6h43m, then the 840 hour extension arrives in bursts
        # until ~init+28h05m. Fire just before the first stage and poll through it; the
        # deadline also clears the previous cycle's last extension files, and the rest
        # of a cycle's extension is swept in one batch by the fire a day later.
        operational_update_cron_job = ReformatCronJob(
            name=f"{cron_job_name_prefix}-update",
            schedule="45 3 * * *",
            pod_active_deadline=timedelta(hours=6),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            # A fire opens by ingesting the whole extension of the previous cycle,
            # the largest single batch of refs it holds.
            memory="16G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=1,
            parallelism=1,
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{cron_job_name_prefix}-validate",
            # The update's fire plus its pod_active_deadline, so the run being
            # validated has always stopped writing.
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
            # A cycle that published nothing is caught here rather than by the
            # completeness check below, which skips append dim positions the store
            # does not reach. Inits are a day apart, so tolerating one would leave a
            # whole day unreported: a cycle is due the moment validation follows it.
            validation.CheckCurrentData(max_delay=timedelta(hours=9, minutes=55)),
            # The newest init holds only its lead times through 384 hours, 105 of 181,
            # when validation fires; the leading tier is that share less a margin. Every
            # older init has its whole 840 hours.
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.55, 1.0)
            ),
            validation.CheckVirtualDecodeHealth(),
        )
