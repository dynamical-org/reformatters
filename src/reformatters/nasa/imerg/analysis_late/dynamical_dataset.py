from datetime import timedelta
from typing import ClassVar

from reformatters.nasa.imerg.analysis_late.region_job import (
    NasaImergAnalysisLateRegionJob,
)
from reformatters.nasa.imerg.analysis_late.template_config import (
    NasaImergAnalysisLateTemplateConfig,
)
from reformatters.nasa.imerg.dynamical_dataset import (
    NasaImergAnalysisMaterializedDataset,
)


class NasaImergAnalysisLateDataset(NasaImergAnalysisMaterializedDataset):
    template_config: NasaImergAnalysisLateTemplateConfig = (
        NasaImergAnalysisLateTemplateConfig()
    )
    region_job_class: type[NasaImergAnalysisLateRegionJob] = (
        NasaImergAnalysisLateRegionJob
    )

    # Late granules publish ~14.7h (p95) after nominal time, landing near :11/:41; run
    # hourly at :44 (~3 min after the :41 landing) to ingest new granules promptly.
    # Re-measure latency with src/scripts/imerg_latency_probe.py if it drifts.
    update_schedule: ClassVar[str] = "44 * * * *"
    validate_schedule: ClassVar[str] = "59 * * * *"
    max_expected_delay: ClassVar[timedelta] = timedelta(hours=18)
