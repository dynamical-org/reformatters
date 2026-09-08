import pandas as pd

from reformatters.common.types import Timestamp
from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.template_config import (
    UcsbChcChirpsAnalysisTemplateConfig,
)


class UcsbChcChirpsAnalysisFinalTemplateConfig(UcsbChcChirpsAnalysisTemplateConfig):
    product: ChirpsProduct = "final"
    append_dim_start: Timestamp = pd.Timestamp("1981-01-01")
