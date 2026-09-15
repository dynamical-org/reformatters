import pandas as pd

from reformatters.common.types import Timestamp
from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.template_config import (
    UcsbChcChirpsAnalysisTemplateConfig,
)


class UcsbChcChirpsAnalysisPreliminaryTemplateConfig(
    UcsbChcChirpsAnalysisTemplateConfig
):
    product: ChirpsProduct = "preliminary"
    append_dim_start: Timestamp = pd.Timestamp("2025-01-01")
