from collections.abc import Sequence
from datetime import timedelta
from functools import partial

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.noaa.hrrr.forecast_virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
    hrrr_virtual_chunk_containers,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar

from .region_job import NoaaHrrrForecast18HourVirtualRegionJob
from .template_config import NoaaHrrrForecast18HourVirtualTemplateConfig


class NoaaHrrrForecast18HourVirtualDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrForecastVirtualSourceFileCoord]
):
    """NOAA HRRR 18-hour virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: NoaaHrrrForecast18HourVirtualTemplateConfig = (
        NoaaHrrrForecast18HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaHrrrForecast18HourVirtualRegionJob] = (
        NoaaHrrrForecast18HourVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=hrrr_virtual_chunk_containers(),
            # Sized per group from ~16.4 bytes/ref (measured on the 48-hour virtual
            # store, which refs the same bucket and URL structure) so full manifests
            # stay within the reader budgets (single-level <= 3 MiB/var, vertical
            # 5-8 MiB); see "Manifest splitting" in docs/virtual_datasets.md.
            # Full-window sizes: single-level 6000 x 19 refs/init ~= 1.8 MiB,
            # pressure 450 x 741 ~= 5.2 MiB, model 400 x 950 ~= 5.9 MiB.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 450,
                    r"^/model_level/": 400,
                    None: 6000,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Run once per hourly cycle just before the first lead times publish
        # (~init+51m), polling through f18 (~init+1h20m..1h40m on S3). The pod exits
        # when the window is fully ingested; the deadline bounds waiting on a file
        # that never publishes and stays under the 1h gap so fires never overlap —
        # a deadline-killed cycle self-heals via the next fires' re-sweep window.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="50 * * * *",
            pod_active_deadline=timedelta(minutes=55),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # Starts in the gap between an update's deadline (:50 + 55m) and the next
            # fire, then may overlap the running update — safe, icechunk reads are
            # snapshot-isolated.
            schedule="48 * * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        # 1h cycle + ~2h publication = ~3h before the latest init is current; 5h
        # tolerates a couple of deadline-deferred cycles without paging.
        return (
            partial(
                validation.check_forecast_current_data,
                max_latest_init_time_age=timedelta(hours=5),
            ),
            # The newest window position is still publishing at validation time, and
            # the one before it is incomplete when the prior update fire was
            # deadline-killed; older positions must be complete.
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.0, 0.0, 1.0)
            ),
            validation.CheckVirtualDecodeHealth(),
        )
