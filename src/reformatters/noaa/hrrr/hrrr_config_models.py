from typing import Literal

from reformatters.common.config_models import DataVar
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


class NoaaHrrrDataVar(DataVar[NoaaHrrrInternalAttrs]):
    pass
