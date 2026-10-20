from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.types import Timestamp
from reformatters.noaa.rtma.grib_header import FieldIdentity


class RtmaInternalAttrs(BaseInternalAttrs):
    grib_element: str
    grib_identity: FieldIdentity
    available_from: Timestamp | None = None


class RtmaDataVar(DataVar[RtmaInternalAttrs]):
    pass
