from reformatters.common.config_models import DataVar
from reformatters.noaa.noaa_config_models import NOAAInternalAttrs


class GFSInternalAttrs(NOAAInternalAttrs):
    pass


class GFSDataVar(DataVar[GFSInternalAttrs]):
    pass
