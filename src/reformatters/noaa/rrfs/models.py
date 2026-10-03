from typing import Literal

import pandas as pd

from reformatters.common.config_models import DataVar
from reformatters.common.types import Timedelta
from reformatters.noaa.models import NoaaInternalAttrs

type RrfsSourceFamily = Literal["2dfld", "prslev", "subh"]


class NoaaRrfsInternalAttrs(NoaaInternalAttrs):
    source_family: RrfsSourceFamily
    minimum_lead_time: Timedelta = pd.Timedelta(0)
    supported_lead_times: tuple[Timedelta, ...] | None = None
    grib_index_selectors: tuple[str, ...] | None = ()


class NoaaRrfsDataVar(DataVar[NoaaRrfsInternalAttrs]):
    def available_at(self, lead_time: Timedelta) -> bool:
        attrs = self.internal_attrs
        return lead_time >= attrs.minimum_lead_time and (
            attrs.supported_lead_times is None
            or lead_time in attrs.supported_lead_times
        )
