from collections.abc import Sequence
from datetime import timedelta

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    HISTORICAL_MANIFEST_INIT_SPLIT,
    GoogleWeathernext2ForecastHistoricalVirtualRegionJob,
    GoogleWeathernext2ForecastVirtualSourceFileCoord,
    weathernext2_virtual_chunk_containers,
)
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2DataVar,
)

from .template_config import (
    GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig,
)


class GoogleWeathernext2ForecastHistoricalVirtualDataset(
    DynamicalDataset[
        GoogleWeathernext2DataVar, GoogleWeathernext2ForecastVirtualSourceFileCoord
    ]
):
    template_config: GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig = (
        GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig()
    )
    region_job_class: type[GoogleWeathernext2ForecastHistoricalVirtualRegionJob] = (
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=weathernext2_virtual_chunk_containers(),
            manifest_split=manifest_append_dim_split(
                split_size=HISTORICAL_MANIFEST_INIT_SPLIT,
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        cron_job_name_prefix = self.dataset_id.replace("weathernext2", "wn2")
        update = ReformatCronJob(
            name=f"{cron_job_name_prefix}-update",
            schedule="0 0 * * *",
            pod_active_deadline=timedelta(hours=6),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.7",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )
        validate = ValidationCronJob(
            name=f"{cron_job_name_prefix}-validate",
            schedule="0 0 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )
        return [update, validate]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
