from typing import Literal

import pandas as pd

from reformatters.common.types import Timedelta, Timestamp
from reformatters.ecmwf.ifs_ens.virtual_template_config import (
    EcmwfIfsEnsForecastVirtualTemplateConfig,
)


class EcmwfIfsEnsForecast6Day01DegreeVirtualTemplateConfig(
    EcmwfIfsEnsForecastVirtualTemplateConfig
):
    resolution: Literal["0p25", "0p1"] = "0p1"
    forecast_days: Literal[6, 15] = 6
    append_dim_frequency: Timedelta = pd.Timedelta("6h")
    append_dim_start: Timestamp = pd.Timestamp("2026-10-21T00")
