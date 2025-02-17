from typing import Literal

from common.config_models import DataVar
from noaa.noaa_config_models import NOAAInternalAttrs

# We pull data from 3 types of source files: `a`, `b` and `s`.
# Selected variables are available in `s` at higher resolution (0.25 vs 0.5 deg)
# but `s` stops after forecast lead time 240h at which point the variable is still in `a` or `b`.
type GEFSFileType = Literal["a", "b", "s+a", "s+b"]


class GEFSInternalAttrs(NOAAInternalAttrs):
    gefs_file_type: GEFSFileType


class GEFSDataVar(DataVar[GEFSInternalAttrs]):
    pass
