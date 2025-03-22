from typing import Literal

from reformatters.common.config_models import DataVar
from reformatters.noaa.noaa_config_models import NOAAInternalAttrs

type HRRRDomain = Literal["alaska", "conus"]

# HRRR Provides 4 datasets corresponding to sets of vertical levels
# - prs: 3D pressure levels
# - nat: Native levels
# - sfc: 2D surface levels
# - subh: 2D surface levels, sub-hourly
type HRRRFileType = Literal["prs", "nat", "sfc", "subh"]


class HRRRInternalAttrs(NOAAInternalAttrs):
    hrrr_file_type: HRRRFileType


class HRRRDataVar(DataVar[HRRRInternalAttrs]):
    pass
