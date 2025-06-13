from reformatters.common.config_models import BaseInternalAttrs, DataVar


class NoaaInternalAttrs(BaseInternalAttrs):
    grib_element: str
    grib_description: str
    grib_index_level: str
    index_position: int
    include_lead_time_suffix: bool = False
    deaccumulate_to_rates: bool = False


class NoaaDataVar(DataVar[NoaaInternalAttrs]):
    pass
