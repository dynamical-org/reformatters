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
