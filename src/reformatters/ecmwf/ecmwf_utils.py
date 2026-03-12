from collections.abc import Sequence

from reformatters.common.types import Timestamp
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar


def all_variables_available(
    data_var_group: Sequence[EcmwfDataVar], init_time: Timestamp
) -> bool:
    """Returns True if all variables in the group are available for the given init time."""
    return all(
        data_var.internal_attrs.date_available is None
        or data_var.internal_attrs.date_available <= init_time
        for data_var in data_var_group
    )


def has_hour_0_values(data_var: EcmwfDataVar) -> bool:
    """Returns True if this variable has a value at lead_time=0h.

    ECMWF avg/accum variables (e.g. total precipitation, radiation) include a 0h
    accumulation of 0 in the GRIB, so they do have hour 0 values. Only "max" and "min"
    step_type variables are absent at lead_time=0h since they represent the extremum
    since the previous post-processing step, which doesn't exist at initialization time.
    """
    if data_var.internal_attrs.hour_0_values_override is not None:
        return data_var.internal_attrs.hour_0_values_override
    return data_var.attrs.step_type not in ("max", "min")
