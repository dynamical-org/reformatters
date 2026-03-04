from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.types import Timedelta


class NoaaInternalAttrs(BaseInternalAttrs):
    grib_element: str
    # Alternative GRIB element names that identify the same variable (e.g. PRMSL vs MSLMA)
    grib_element_alternatives: tuple[str, ...] = ()
    grib_description: str
    grib_index_level: str
    index_position: int
    include_lead_time_suffix: bool = False
    # for step_type != "instant"
    window_reset_frequency: Timedelta | None = None
    # True for running-total accumulations (e.g. ASNOW) that never reset within a forecast.
    # Used to generate the correct "0-N hour/day acc fcst" index strings even when window_reset_frequency is set.
    grib_lead_time_is_running_total: bool = False


class NoaaDataVar(DataVar[NoaaInternalAttrs]):
    pass
