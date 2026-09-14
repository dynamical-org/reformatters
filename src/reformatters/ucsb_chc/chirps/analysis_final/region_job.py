from datetime import timedelta

from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisMaterializedRegionJob,
)


class UcsbChcChirpsAnalysisFinalRegionJob(UcsbChcChirpsAnalysisMaterializedRegionJob):
    product: ChirpsProduct = "final"
    # Monthly batches arrive in the third week of the following month, so the first
    # day of a month has the longest publication delay.
    expected_missing_window = timedelta(days=60)
