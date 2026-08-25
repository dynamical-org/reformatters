from typing import Literal

from reformatters.common.config_models import DataVar
from reformatters.common.types import Timestamp
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


class NoaaHrrrDataVar(DataVar[NoaaHrrrInternalAttrs]):
    pass
