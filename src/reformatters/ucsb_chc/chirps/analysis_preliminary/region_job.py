from datetime import timedelta
from typing import ClassVar

import pandas as pd

from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisMaterializedRegionJob,
)


class UcsbChcChirpsAnalysisPreliminaryRegionJob(
    UcsbChcChirpsAnalysisMaterializedRegionJob
):
    product: ChirpsProduct = "preliminary"
    # The preliminary product publishes one pentad at a time, two days after the
    # pentad ends.
    expected_missing_window = timedelta(days=10)
    known_missing_days: ClassVar[frozenset[pd.Timestamp]] = frozenset(
        pd.date_range("2025-02-26", "2025-02-28")
    ) | frozenset(pd.date_range("2025-03-26", "2025-03-31"))
