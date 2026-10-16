from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.noaa.rtma.grib_header import FieldIdentity


class RtmaInternalAttrs(BaseInternalAttrs):
    grib_element: str
    grib_identity: FieldIdentity


class RtmaDataVar(DataVar[RtmaInternalAttrs]):
    pass
