from reformatters.common.config_models import DataVar
from reformatters.noaa.noaa_config_models import NOAAInternalAttrs


class HRRRInternalAttrs(NOAAInternalAttrs):
    pass


class HRRRDataVar(DataVar[HRRRInternalAttrs]):
    pass
