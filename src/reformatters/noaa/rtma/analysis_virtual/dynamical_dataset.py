from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

import icechunk
from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.noaa.rtma.rtma_config_models import RtmaDataVar

from .region_job import (
    SOURCE_PREFIX,
    SOURCE_REGION,
    NoaaRtmaAnalysisVirtualRegionJob,
    NoaaRtmaAnalysisVirtualSourceFileCoord,
)
from .template_config import NoaaRtmaAnalysisVirtualTemplateConfig


class NoaaRtmaAnalysisVirtualDataset(
    DynamicalDataset[RtmaDataVar, NoaaRtmaAnalysisVirtualSourceFileCoord]
):
    update_schedule: ClassVar[str] = "26 * * * *"
    update_deadline: ClassVar[timedelta] = timedelta(hours=1, minutes=2, seconds=9)
    update_cron_suspended: ClassVar[bool] = False
    max_delay: ClassVar[timedelta] = timedelta(minutes=74)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(seconds=43)

    template_config: NoaaRtmaAnalysisVirtualTemplateConfig = (
        NoaaRtmaAnalysisVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRtmaAnalysisVirtualRegionJob] = (
        NoaaRtmaAnalysisVirtualRegionJob
    )
    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=(
                icechunk.VirtualChunkContainer(
                    url_prefix=SOURCE_PREFIX,
                    store=icechunk.s3_store(region=SOURCE_REGION, anonymous=True),
                ),
            ),
            manifest_split=manifest_append_dim_split(split_size=30000, dim="time"),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule=self.update_schedule,
                concurrency_policy="Allow",
                active_deadline=self.update_deadline,
                pod_active_deadline=self.update_deadline,
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1.7",
                memory="7G",
                secret_names=self.store_factory.k8s_secret_names(),
                workers_total=1,
                parallelism=1,
                suspend=self.update_cron_suspended,
            )
        ]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=self.max_delay),
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        )
