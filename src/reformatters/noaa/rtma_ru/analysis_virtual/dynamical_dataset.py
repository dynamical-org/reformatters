from datetime import timedelta
from typing import ClassVar

from reformatters.noaa.rtma.analysis_virtual.dynamical_dataset import (
    NoaaRtmaAnalysisVirtualDataset,
)

from .region_job import NoaaRtmaRuAnalysisVirtualRegionJob
from .template_config import NoaaRtmaRuAnalysisVirtualTemplateConfig


class NoaaRtmaRuAnalysisVirtualDataset(NoaaRtmaAnalysisVirtualDataset):
    update_schedule: ClassVar[str] = "6,21,36,51 * * * *"
    update_deadline: ClassVar[timedelta] = timedelta(minutes=17, seconds=9)
    update_cron_suspended: ClassVar[bool] = False
    max_delay: ClassVar[timedelta] = timedelta(minutes=43)

    template_config: NoaaRtmaRuAnalysisVirtualTemplateConfig = (
        NoaaRtmaRuAnalysisVirtualTemplateConfig()
    )
    region_job_class: type[NoaaRtmaRuAnalysisVirtualRegionJob] = (
        NoaaRtmaRuAnalysisVirtualRegionJob
    )
