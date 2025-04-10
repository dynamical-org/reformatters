from typing import Literal

import pandas as pd

from reformatters.common.config_models import DataVar
from reformatters.noaa.noaa_config_models import NOAAInternalAttrs

# We pull data from 3 types of source files: `a`, `b` and `s`.
# Selected variables are available in `s` at higher resolution (0.25 vs 0.5 deg)
# but `s` stops after forecast lead time 240h at which point the variable is still in `a` or `b`.
# `s+b-b22` is the same as `s+b` when init time >= 2022-10-18T12 and `b` before.
type GEFSFileType = Literal["a", "b", "s+a", "s+b", "s+b-b22"]
GEFS_S_FILE_MAX = pd.Timedelta(hours=240)
GEFS_B22_TRANSITION_DATE = pd.Timestamp("2022-10-18T12:00")


class GEFSInternalAttrs(NOAAInternalAttrs):
    gefs_file_type: GEFSFileType


class GEFSDataVar(DataVar[GEFSInternalAttrs]):
    pass
