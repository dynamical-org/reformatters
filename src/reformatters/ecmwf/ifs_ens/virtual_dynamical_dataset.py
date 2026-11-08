from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split

from .virtual_region_job import (
    EcmwfIfsEnsForecastVirtualRegionJob,
    EcmwfIfsEnsForecastVirtualSourceFileCoord,
    ifs_ens_virtual_chunk_containers,
)
from .virtual_template_config import (
    EcmwfIfsEnsForecastVirtualTemplateConfig,
    EcmwfIfsEnsVirtualDataVar,
)
from .virtual_validation import (
    EcmwfIfsEnsVirtualDecodeHealth,
    EcmwfIfsEnsVirtualManifestCompleteness,
)


class EcmwfIfsEnsForecastVirtualDataset(
    DynamicalDataset[
        EcmwfIfsEnsVirtualDataVar, EcmwfIfsEnsForecastVirtualSourceFileCoord
    ]
):
    template_config: EcmwfIfsEnsForecastVirtualTemplateConfig
    region_job_class: type[EcmwfIfsEnsForecastVirtualRegionJob]
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=7)
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda data: IcechunkVirtualConfig(
            containers=ifs_ens_virtual_chunk_containers(
                fine_resolution=data["template_config"].resolution == "0p1"
            ),
            # At 85 leads: 40 x 85 x 51 x 16 bytes = 2.65 MiB per root array;
            # 6 x 85 x 51 x 14 x 16 bytes = 5.56 MiB per pressure array.
            manifest_split=manifest_append_dim_split(
                split_size={r"^/pressure_level/": 6, None: 40}, dim="init_time"
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        short = self.template_config.forecast_days == 6
        fine = self.template_config.resolution == "0p1"
        hour = (8 if fine else 6) if short else (9 if fine else 7)
        hours = ",".join(
            str((hour + offset) % 24) for offset in range(0, 24, 6 if short else 12)
        )
        deadline = timedelta(hours=6 if short else 12, minutes=8)
        return [
            ReformatCronJob(
                name=f"{self.dataset_id.replace('-degree', '')}-update",
                schedule=f"{12 if short else 19} {hours} * * *",
                concurrency_policy="Allow",
                suspend=True,
                pod_active_deadline=deadline,
                active_deadline=deadline,
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
        # Control files make the position visible before perturbed files arrive.
        delay = timedelta(
            hours=9 if self.template_config.resolution == "0p1" else 7, minutes=42
        )
        return (
            validation.CheckCurrentData(max_delay=delay),
            EcmwfIfsEnsVirtualManifestCompleteness(),
            EcmwfIfsEnsVirtualDecodeHealth(
                sampled_leads=3, sampled_levels=2, max_workers=2
            ),
        )
