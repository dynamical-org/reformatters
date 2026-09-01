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
            # Four days of 6 hourly inits. Every array holds one ref per (lead time,
            # ensemble member), so a full manifest is 16 x 81 x 31 = 40176 refs, which
            # measures 0.68 MiB (17.8 bytes/ref) on this dataset's own manifests: well
            # inside the 3 MiB reader budget and far above the 1000 refs zstd location
            # compression needs. Splitting finer would cut per-commit flush cost but
            # multiply the manifest count all 38 arrays share; see "Manifest splitting"
            # in docs/virtual_datasets.md for the cost model.
            manifest_split=manifest_append_dim_split(split_size=16, dim="init_time"),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # The dataset id plus "-validate" exceeds the 52 character cron job name limit,
        # so the resolution takes its source file spelling.
        cron_job_name_prefix = self.dataset_id.replace("-0-25-degree", "-0p25")
        # The whole run publishes in one burst: f000 lands ~init+3h47m and the last
        # member's f240 ~init+5h37m. Fire just before the burst starts and poll through
        # it; the deadline leaves ~35 minutes of slack past the observed end and still
        # ends well before the next cycle's fire.
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
            # The update's fire plus its pod_active_deadline, so the run being
            # validated has always stopped writing.
            schedule="15 6,12,18,0 * * *",
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
            # The newest init is 6h15m old when validation fires, so 12h adds a
            # cycle of slack: a run that rolls its last files to the next fire still
            # passes, while two cycles with nothing ingested fail.
            validation.CheckCurrentData(max_delay=timedelta(hours=12)),
            # The newest init is fully published ~38 minutes before validation fires, so
            # its leading tier only absorbs a slow cycle: 0.95 is the ~2500 files of a
            # run less the last four lead times of every member, which the source lays
            # down in its final ~20 minutes. A cycle that published nothing, or a run
            # missing a member or a variable's whole lead range, still fails.
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.95, 1.0)
            ),
            validation.CheckVirtualDecodeHealth(),
        )
