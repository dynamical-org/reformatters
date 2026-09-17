from collections.abc import Sequence
from typing import Literal

import pandas as pd

from reformatters.common.config_models import DataVar
from reformatters.common.iterating import group_by
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
    # Time from which the source's hour-0 field is unusable while its later leads are fine,
    # so an analysis reads the previous init's 1 hour lead instead. Analysis only.
    # Access via data_var.analysis_lead_time(time), not directly.
    analysis_hour_0_unusable_from: Timestamp | None = None


class NoaaHrrrDataVar(DataVar[NoaaHrrrInternalAttrs]):
    def analysis_lead_time(self, time: Timestamp) -> Timedelta:
        """The lead time an analysis at `time` reads this variable at; its init is that long before `time`."""
        unusable_from = self.internal_attrs.analysis_hour_0_unusable_from
        if self.has_hour_0_values() and (unusable_from is None or time < unusable_from):
            return pd.Timedelta("0h")
        return pd.Timedelta("1h")


def analysis_source_file_var_groups(
    data_vars: Sequence[NoaaHrrrDataVar], time: Timestamp
) -> Sequence[Sequence[NoaaHrrrDataVar]]:
    """Variables an analysis at `time` reads from the same source file: same file type and lead time."""
    return group_by(
        data_vars,
        lambda v: (v.internal_attrs.hrrr_file_type, v.analysis_lead_time(time)),
    )
