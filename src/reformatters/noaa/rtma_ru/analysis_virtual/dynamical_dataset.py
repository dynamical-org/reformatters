from datetime import timedelta
from typing import ClassVar

from reformatters.common.kubernetes import ConcurrencyPolicy
from reformatters.noaa.rtma.analysis_virtual.dynamical_dataset import (
    NoaaRtmaAnalysisVirtualDataset,
)

from .region_job import NoaaRtmaRuAnalysisVirtualRegionJob
from .template_config import NoaaRtmaRuAnalysisVirtualTemplateConfig


class NoaaRtmaRuAnalysisVirtualDataset(NoaaRtmaAnalysisVirtualDataset):
    concurrency_policy: ClassVar[ConcurrencyPolicy] = "Replace"
    update_schedule: ClassVar[str] = "*/15 * * * *"
    update_deadline: ClassVar[timedelta] = timedelta(minutes=14)
    update_cron_suspended: ClassVar[bool] = True
    max_delay: ClassVar[timedelta] = timedelta(minutes=45)
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=2)

    template_config: NoaaRtmaRuAnalysisVirtualTemplateConfig = (
        NoaaRtmaRuAnalysisVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRtmaRuAnalysisVirtualRegionJob] = (
        NoaaRtmaRuAnalysisVirtualRegionJob
    )
