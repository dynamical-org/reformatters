from typing import ClassVar

from reformatters.ucsb_chc.chirps.analysis_final.region_job import (
    UcsbChcChirpsAnalysisFinalRegionJob,
)
from reformatters.ucsb_chc.chirps.analysis_final.template_config import (
    UcsbChcChirpsAnalysisFinalTemplateConfig,
)
from reformatters.ucsb_chc.chirps.dynamical_dataset import (
    UcsbChcChirpsAnalysisMaterializedDataset,
)


class UcsbChcChirpsAnalysisFinalDataset(UcsbChcChirpsAnalysisMaterializedDataset):
    template_config: UcsbChcChirpsAnalysisFinalTemplateConfig = (
        UcsbChcChirpsAnalysisFinalTemplateConfig()
    )
    region_job_class: type[UcsbChcChirpsAnalysisFinalRegionJob] = (
        UcsbChcChirpsAnalysisFinalRegionJob
    )

    update_schedule: ClassVar[str] = "0 23 * * *"
    validate_schedule: ClassVar[str] = "0 0 * * *"
