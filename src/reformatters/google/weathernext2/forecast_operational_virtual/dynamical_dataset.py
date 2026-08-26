from collections.abc import Sequence
from datetime import timedelta

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT,
    OPERATIONAL_ROOT_MANIFEST_INIT_SPLIT,
    GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    GoogleWeathernext2ForecastVirtualSourceFileCoord,
    weathernext2_virtual_chunk_containers,
)
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2DataVar,
)

from .template_config import (
    GoogleWeathernext2ForecastOperationalVirtualTemplateConfig,
)


class GoogleWeathernext2ForecastOperationalVirtualDataset(
    DynamicalDataset[
        GoogleWeathernext2DataVar, GoogleWeathernext2ForecastVirtualSourceFileCoord
    ]
):
    template_config: GoogleWeathernext2ForecastOperationalVirtualTemplateConfig = (
        GoogleWeathernext2ForecastOperationalVirtualTemplateConfig()
    )
    region_job_class: type[GoogleWeathernext2ForecastOperationalVirtualRegionJob] = (
        GoogleWeathernext2ForecastOperationalVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=weathernext2_virtual_chunk_containers(),
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT,
                    None: OPERATIONAL_ROOT_MANIFEST_INIT_SPLIT,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        suspend = True
        update = ReformatCronJob(
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
        validate = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="55 1,7,13,19 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=suspend,
        )
        return [update, validate]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=60)),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
