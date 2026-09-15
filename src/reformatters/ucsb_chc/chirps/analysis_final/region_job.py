from datetime import timedelta

from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisMaterializedRegionJob,
)


class UcsbChcChirpsAnalysisFinalRegionJob(UcsbChcChirpsAnalysisMaterializedRegionJob):
    product: ChirpsProduct = "final"
    expected_unavailable_window = timedelta(days=60)
