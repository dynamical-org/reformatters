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

from .region_job import NoaaGefsForecast10Day025DegreeVirtualRegionJob
from .template_config import NoaaGefsForecast10Day025DegreeVirtualTemplateConfig


class NoaaGefsForecast10Day025DegreeVirtualDataset(
    DynamicalDataset[NoaaGefsVirtualDataVar, NoaaGefsForecastVirtualSourceFileCoord]
):
    """NOAA GEFS 10 day 0.25 degree virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: NoaaGefsForecast10Day025DegreeVirtualTemplateConfig = (
        NoaaGefsForecast10Day025DegreeVirtualTemplateConfig()
    )
    region_job_class: type[NoaaGefsForecast10Day025DegreeVirtualRegionJob] = (
        NoaaGefsForecast10Day025DegreeVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=gefs_virtual_chunk_containers(),
            # Four days of 6 hourly inits.
            manifest_split=manifest_append_dim_split(split_size=16, dim="init_time"),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # The dataset id plus "-validate" exceeds the 52 character cron job name limit.
        cron_job_name_prefix = self.dataset_id.replace("-0-25-degree", "-0-25")
        # A run publishes ~init+3h47m through ~init+5h37m.
        # Fire just before the first files become available and stop 30m after expected completion.
        operational_update_cron_job = ReformatCronJob(
            name=f"{cron_job_name_prefix}-update",
            schedule="45 3,9,15,21 * * *",
            pod_active_deadline=timedelta(hours=2, minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=1,
            parallelism=1,
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{cron_job_name_prefix}-validate",
            # The update's fire plus its pod_active_deadline
            schedule="25 6,12,18,0 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=6, minutes=20)),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
