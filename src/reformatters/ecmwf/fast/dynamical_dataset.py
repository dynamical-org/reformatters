from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from pydantic import Field

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
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=2, seconds=30)
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=virtual_containers(),
            source_secret_names={SOURCE_PREFIX: SOURCE_SECRET},
            manifest_split=manifest_append_dim_split(split_size=16, dim="init_time"),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        minute = 8 if self.template_config.model == "ifs" else 34
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule=f"{minute} 5,11,17,23 * * *",
                suspend=True,
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
        delay = (
            timedelta(hours=5, minutes=36)
            if self.template_config.model == "ifs"
            else timedelta(hours=6, minutes=2)
        )
        return (
            validation.CheckCurrentData(max_delay=delay),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(sampled_leads=3, max_workers=3),
        )
