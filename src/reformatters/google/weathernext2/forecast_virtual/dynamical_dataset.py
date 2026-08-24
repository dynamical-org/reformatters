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
    PRESSURE_MANIFEST_INIT_SPLIT,
    ROOT_MANIFEST_INIT_SPLIT,
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
            # Keep each active manifest near 2 MB and bound an append to 14 manifests.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": PRESSURE_MANIFEST_INIT_SPLIT,
                    None: ROOT_MANIFEST_INIT_SPLIT,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Remove after backfilling to run operational updates and validation.
        suspend = True
        # Each fire publishes the forecast planes whose valid times crossed the strict
        # 48-hour lag boundary, while the 18-day update window catches every lead.
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
            # After each update fire and its 30 minute deadline.
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
        # With 6-hourly inits and a 6-hour minimum lead, the newest eligible init is
        # almost 54 hours old; allow one cycle of scheduling slack.
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=60)),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
