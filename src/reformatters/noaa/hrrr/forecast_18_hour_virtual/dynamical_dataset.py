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
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
    hrrr_virtual_chunk_containers,
)

from .region_job import NoaaHrrrForecast18HourVirtualRegionJob
from .template_config import NoaaHrrrForecast18HourVirtualTemplateConfig


class NoaaHrrrForecast18HourVirtualDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrForecastVirtualSourceFileCoord]
):
    template_config: NoaaHrrrForecast18HourVirtualTemplateConfig = (
        NoaaHrrrForecast18HourVirtualTemplateConfig()
    )
    region_job_class: type[NoaaHrrrForecast18HourVirtualRegionJob] = (
        NoaaHrrrForecast18HourVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=hrrr_virtual_chunk_containers(),
            # Scale the 48-hour product's splits by 2.5 to keep full manifest
            # byte sizes comparable across the two forecast lengths.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 225,
                    r"^/model_level/": 200,
                    None: 1500,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Race the current init: f00 arrives near :51 and f18 normally by init + 86m.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="50 * * * *",
            pod_active_deadline=timedelta(minutes=59),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # The update's fire plus its pod_active_deadline, so the run being
            # validated has always stopped writing.
            schedule="49 * * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        return (
            partial(
                validation.check_forecast_current_data,
                max_latest_init_time_age=timedelta(hours=2),
            ),
            # Newest ingested init: the run that just ended may have deferred late files
            # to the next fire, but f00 lands an hour before its poll deadline, so 5%
            # (3 of 57 files, one lead's worth) separates a deferral from a cycle that
            # published nothing.
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.05, 1.0)
            ),
            validation.CheckVirtualDecodeHealth(),
        )
