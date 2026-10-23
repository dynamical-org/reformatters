from datetime import timedelta
from typing import ClassVar

import icechunk
from pydantic import Field

from reformatters.common.storage import IcechunkVirtualConfig, manifest_append_dim_split
from reformatters.noaa.rtma.analysis_virtual.dynamical_dataset import (
    NoaaRtmaAnalysisVirtualDataset,
)
from reformatters.noaa.rtma.analysis_virtual.region_job import SOURCE_REGION

from .region_job import SOURCE_PREFIX, NoaaUrmaAnalysisVirtualRegionJob
from .template_config import NoaaUrmaAnalysisVirtualTemplateConfig


class NoaaUrmaAnalysisVirtualDataset(NoaaRtmaAnalysisVirtualDataset):
    update_deadline: ClassVar[timedelta] = timedelta(minutes=59)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=2)
    max_delay: ClassVar[timedelta] = timedelta(hours=7, minutes=40)
    update_cron_suspended: ClassVar[bool] = True

    template_config: NoaaUrmaAnalysisVirtualTemplateConfig = (
        NoaaUrmaAnalysisVirtualTemplateConfig()
    )
    region_job_class: type[NoaaUrmaAnalysisVirtualRegionJob] = (
        NoaaUrmaAnalysisVirtualRegionJob
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
