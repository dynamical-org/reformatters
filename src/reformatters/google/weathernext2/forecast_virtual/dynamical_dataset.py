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

from .region_job import (
    GoogleWeathernext2ForecastVirtualRegionJob,
    GoogleWeathernext2ForecastVirtualSourceFileCoord,
    weathernext2_virtual_chunk_containers,
)
from .template_config import (
    GoogleWeathernext2DataVar,
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)


class GoogleWeathernext2ForecastVirtualDataset(
    DynamicalDataset[
        GoogleWeathernext2DataVar, GoogleWeathernext2ForecastVirtualSourceFileCoord
    ]
):
    """Google WeatherNext 2 virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: GoogleWeathernext2ForecastVirtualTemplateConfig = (
        GoogleWeathernext2ForecastVirtualTemplateConfig()
    )
    region_job_class: type[GoogleWeathernext2ForecastVirtualRegionJob] = (
        GoogleWeathernext2ForecastVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=weathernext2_virtual_chunk_containers(),
            # Sized for operational commit latency: active-window manifest bytes bound
            # per-commit flush cost. Full-window sizes at ~16.4 bytes/ref: root
            # 600 x 60 refs/init ~= 0.6 MiB, pressure 200 x 780 (60 leads x 13 levels)
            # ~= 2.4 MiB; see "Manifest splitting" in docs/virtual_datasets.md for the
            # cost model.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 200,
                    None: 600,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Remove after backfilling to run operational updates and validation.
        suspend = True
        # Run once per 6h cycle just after the store publishes: all 60 leads land in a
        # ~3-4 minute burst ~init+6h15m and the success marker follows at init+6h20m to
        # 6h50m. Fire at init+6h55m, past the late end of that range, so the cycle is
        # ingested on the first tick; the 30 minute deadline bounds the poll for the next
        # cycle's slot (which publication lag puts ~6h out) and keeps fires from
        # overlapping.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="55 0,6,12,18 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.7",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=suspend,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # After each update (init+6h55m) + its 30 minute deadline.
            schedule="55 1,7,13,19 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=suspend,
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        # Validation fires at init+7h55m, after the update's fire and deadline; the newest
        # ingested init is then 7h55m old, so 9h leaves an hour of cron/pod start slack.
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=9)),
            # A store is published behind a success marker written last, so an ingested
            # init is a whole one. Positions past the store's extent are skipped, which
            # covers the window's newest, not-yet-published cycle.
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
