from datetime import timedelta
from typing import ClassVar

from reformatters.nasa.imerg.analysis_early.region_job import (
    NasaImergAnalysisEarlyRegionJob,
)
from reformatters.nasa.imerg.analysis_early.template_config import (
    NasaImergAnalysisEarlyTemplateConfig,
)
from reformatters.nasa.imerg.dynamical_dataset import (
    NasaImergAnalysisMaterializedDataset,
)


class NasaImergAnalysisEarlyDataset(NasaImergAnalysisMaterializedDataset):
    template_config: NasaImergAnalysisEarlyTemplateConfig = (
        NasaImergAnalysisEarlyTemplateConfig()
    )
    region_job_class: type[NasaImergAnalysisEarlyRegionJob] = (
        NasaImergAnalysisEarlyRegionJob
    )

    # Early granules publish ~4.6h (p95) after nominal time, landing near :05/:35; run
    # hourly at :38 (~3 min after the :35 landing) to ingest new granules promptly.
    # Re-measure latency with src/scripts/imerg_latency_probe.py if it drifts.
    update_schedule: ClassVar[str] = "38 * * * *"
    validate_schedule: ClassVar[str] = "53 * * * *"
    max_expected_delay: ClassVar[timedelta] = timedelta(hours=8)
