from typing import Literal

import pandas as pd

from reformatters.common.config_models import DataVar
from reformatters.common.types import Timedelta, Timestamp
from reformatters.noaa.models import NoaaInternalAttrs

type NoaaHrrrDomain = Literal["alaska", "conus"]

# HRRR Provides 4 datasets corresponding to sets of vertical levels
# - prs: 3D pressure levels
# - nat: Native levels
# - sfc: 2D surface levels
# - subh: 2D surface levels, sub-hourly
type NoaaHrrrFileType = Literal["prs", "nat", "sfc", "subh"]


class NoaaHrrrInternalAttrs(NoaaInternalAttrs):
    hrrr_file_type: NoaaHrrrFileType
    # Multiply raw values by this factor after reading (e.g. 0.01 to convert percent to fraction)
    scale_factor: float | None = None
    # Time before which an analysis holds nothing a reader can use, so it reads no source
    # file there and returns NaN. Analysis only: a field can be unusable at the hour an
    # analysis takes while the longer leads a forecast also carries are fine.
    analysis_usable_from: Timestamp | None = None
    # Analysis only: the source's hour-0 field is unusable while its later leads are fine,
    # so an analysis reads the previous init's 1 hour lead instead. A forecast still serves
    # hour 0 as published. HRRR hour 0 carries scattered pixels of collapsed 2 m moisture:
    # dew point down to -81 C, relative humidity at 1 %, specific humidity at 0.
    # Access via data_var.analysis_lead_time(), not directly.
    analysis_hour_0_unusable: bool = False


class NoaaHrrrDataVar(DataVar[NoaaHrrrInternalAttrs]):
    def analysis_lead_time(self) -> Timedelta:
        """The lead time an analysis reads this variable at; its init is that long before the analysis time."""
        if (
            self.has_hour_0_values()
            and not self.internal_attrs.analysis_hour_0_unusable
        ):
            return pd.Timedelta("0h")
        return pd.Timedelta("1h")
