from datetime import timedelta
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
    update_deadline: ClassVar[timedelta] = timedelta(minutes=60)
    # The final product publishes a whole month at once, about two weeks after the
    # month ends, so the first day of a month is the last one to arrive.
    max_expected_delay: ClassVar[timedelta] = timedelta(days=60)
