from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.ecmwf.ecmwf_virtual_source import ecmwf_virtual_chunk_containers

from .virtual_region_job import (
    EcmwfIfsEnsForecastVirtualRegionJob,
    EcmwfIfsEnsForecastVirtualSourceFileCoord,
)
from .virtual_template_config import (
    EcmwfIfsEnsForecastVirtualTemplateConfig,
    EcmwfIfsEnsVirtualDataVar,
)


class EcmwfIfsEnsForecastVirtualDataset(
    DynamicalDataset[
        EcmwfIfsEnsVirtualDataVar, EcmwfIfsEnsForecastVirtualSourceFileCoord
    ]
):
    template_config: EcmwfIfsEnsForecastVirtualTemplateConfig
    region_job_class: type[EcmwfIfsEnsForecastVirtualRegionJob]
    schedule: ClassVar[str]
    deadline: ClassVar[timedelta]
    # Delay tracks control files which make a position visible before perturbed files arrive.
    max_delay: ClassVar[timedelta]
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=5)
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda data: IcechunkVirtualConfig(
            containers=ecmwf_virtual_chunk_containers(
                s3_fallback_before=data["region_job_class"].s3_fallback_before
            ),
            # At 85 leads: 40 x 85 x 51 x 16 bytes = 2.65 MiB per root array;
            # 6 x 85 x 51 x 14 x 16 bytes = 5.56 MiB per pressure array.
            manifest_split=manifest_append_dim_split(
                split_size={r"^/pressure_level/": 6, None: 40}, dim="init_time"
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return [
            ReformatCronJob(
                name=f"{self.dataset_id.replace('-degree', '')}-update",
                schedule=self.schedule,
                concurrency_policy="Allow",
                pod_active_deadline=self.deadline,
                active_deadline=self.deadline,
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1.7",
                memory="7G",
                secret_names=self.store_factory.k8s_secret_names(),
                workers_total=1,
                parallelism=1,
            )
        ]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=self.max_delay),
            validation.CheckVirtualManifestCompleteness(),
            # Bounds decoded memory: each sampled field is up to 52 MB at 0.1 degree.
            validation.CheckVirtualDecodeHealth(
                sampled_leads=3, sampled_levels=2, max_workers=2
            ),
        )
