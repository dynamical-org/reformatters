from typing import Literal

import pandas as pd

from reformatters.common.config_models import DataVar
from reformatters.common.types import Timedelta
from reformatters.noaa.models import NoaaInternalAttrs

type ProductFamily = Literal[
    "mean", "sprd", "prob", "pmmn", "lpmm", "avrg", "eas", "ffri"
]


class NoaaRefsInternalAttrs(NoaaInternalAttrs):
    source_families: tuple[ProductFamily, ...]
    supported_lead_times: tuple[Timedelta, ...]


class NoaaRefsDataVar(DataVar[NoaaRefsInternalAttrs]):
    has_statistic: bool = False

    def available_at(self, lead_time: Timedelta) -> bool:
        return lead_time in self.internal_attrs.supported_lead_times


def supported_leads(duration: int) -> tuple[Timedelta, ...]:
    return tuple(
        pd.Timedelta(hours=hour)
        for hour in range(max(1, duration), 61, 1 if duration <= 1 else 3)
    )
