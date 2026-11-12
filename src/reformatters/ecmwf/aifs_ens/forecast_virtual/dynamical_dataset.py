from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.ecmwf.ecmwf_virtual_source import ecmwf_virtual_chunk_containers

from .region_job import (
    EcmwfAifsEnsForecastVirtualRegionJob,
    EcmwfAifsEnsForecastVirtualSourceFileCoord,
)
from .template_config import (
    EcmwfAifsEnsForecastVirtualTemplateConfig,
    EcmwfAifsEnsVirtualDataVar,
)


class EcmwfAifsEnsForecastVirtualDataset(
    DynamicalDataset[
        EcmwfAifsEnsVirtualDataVar, EcmwfAifsEnsForecastVirtualSourceFileCoord
    ]
):
    template_config: EcmwfAifsEnsForecastVirtualTemplateConfig = (
        EcmwfAifsEnsForecastVirtualTemplateConfig()
    )
    region_job_class: type[EcmwfAifsEnsForecastVirtualRegionJob] = (
        EcmwfAifsEnsForecastVirtualRegionJob
    )
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=5)
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=ecmwf_virtual_chunk_containers(s3_fallback_before=None),
            manifest_split=manifest_append_dim_split(
                split_size={r"^/pressure_level/": 8, None: 60}, dim="init_time"
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule="5 5,11,17,23 * * *",
                concurrency_policy="Allow",
                pod_active_deadline=timedelta(hours=6, minutes=6),
                active_deadline=timedelta(hours=6, minutes=6),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1.7",
                memory="7G",
                secret_names=self.store_factory.k8s_secret_names(),
                workers_total=1,
                parallelism=1,
                suspend=True,
            )
        ]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            # Hand value from the GCS source regime starting 2026-10-27T12Z; measure with --days after it.
            validation.CheckCurrentData(max_delay=timedelta(hours=8, minutes=13)),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(
                sampled_leads=3, sampled_levels=2, max_workers=2
            ),
        )
