import pandas as pd

from reformatters.common.types import Timestamp
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)


class EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig(
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig
):
    append_dim_start: Timestamp = pd.Timestamp("2024-04-01T00:00")
