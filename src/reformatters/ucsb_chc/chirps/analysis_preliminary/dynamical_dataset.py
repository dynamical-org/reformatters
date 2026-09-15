from typing import ClassVar

from reformatters.ucsb_chc.chirps.analysis_preliminary.region_job import (
    UcsbChcChirpsAnalysisPreliminaryRegionJob,
)
from reformatters.ucsb_chc.chirps.analysis_preliminary.template_config import (
    UcsbChcChirpsAnalysisPreliminaryTemplateConfig,
)
from reformatters.ucsb_chc.chirps.dynamical_dataset import (
    UcsbChcChirpsAnalysisMaterializedDataset,
)


class UcsbChcChirpsAnalysisPreliminaryDataset(UcsbChcChirpsAnalysisMaterializedDataset):
    template_config: UcsbChcChirpsAnalysisPreliminaryTemplateConfig = (
        UcsbChcChirpsAnalysisPreliminaryTemplateConfig()
    )
    region_job_class: type[UcsbChcChirpsAnalysisPreliminaryRegionJob] = (
        UcsbChcChirpsAnalysisPreliminaryRegionJob
    )

    update_schedule: ClassVar[str] = "0 19 * * *"
    validate_schedule: ClassVar[str] = "0 20 * * *"
