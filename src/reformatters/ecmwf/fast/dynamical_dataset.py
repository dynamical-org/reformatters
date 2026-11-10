from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar, assert_never

from pydantic import Field, PositiveInt

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar

from .region_job import EcmwfFastRegionJob
from .source import (
    SOURCE_PREFIX,
    SOURCE_SECRET,
    EcmwfFastSourceFileCoord,
    virtual_containers,
)
from .template_config import EcmwfFastTemplateConfig


class EcmwfFastDataset(DynamicalDataset[EcmwfDataVar, EcmwfFastSourceFileCoord]):
    template_config: EcmwfFastTemplateConfig
    region_job_class: type[EcmwfFastRegionJob]
    rolling_positions: PositiveInt | None = 12
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=2, seconds=30)
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=virtual_containers(),
            source_secret_names={SOURCE_PREFIX: SOURCE_SECRET},
            manifest_split=manifest_append_dim_split(split_size=1, dim="init_time"),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        match self.template_config.model:
            case "ifs":
                minute = 8
            case "aifs":
                minute = 34
            case unexpected:
                assert_never(unexpected)
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule=f"{minute} 5,11,17,23 * * *",
                concurrency_policy="Allow",
                pod_active_deadline=timedelta(hours=6, minutes=3, seconds=30),
                active_deadline=timedelta(hours=6, minutes=3, seconds=30),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1.7",
                memory="7G",
                ephemeral_storage="20G",
                workers_total=1,
                parallelism=1,
                secret_names=self.store_factory.k8s_secret_names(),
            )
        ]

    def validators(self) -> Sequence[validation.Validator]:
        match self.template_config.model:
            case "ifs":
                delay = timedelta(hours=5, minutes=36)
            case "aifs":
                delay = timedelta(hours=6, minutes=2)
            case unexpected:
                assert_never(unexpected)
        return (
            validation.CheckCurrentData(max_delay=delay),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(
                sampled_leads=3,
                max_workers=3,
                reference_presence=lambda var_path, _out_loc: {
                    None: var_path
                    not in ("pressure_surface", "dew_point_temperature_2m")
                },
            ),
        )
