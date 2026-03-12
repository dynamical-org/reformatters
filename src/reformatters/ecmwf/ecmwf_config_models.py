from collections.abc import Sequence
from typing import Literal

import pandas as pd

from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.iterating import item
from reformatters.common.types import Timedelta, Timestamp


class EcmwfInternalAttrs(BaseInternalAttrs):
    """
    Variable specific attributes used internally to drive processing.
    Not written to the dataset.

    Functional fields:
        window_reset_frequency: for resetting deaccumulation windows
        grib_index_param: The short name of the param as it exists in the index file. Does not map to any names in the grib.
        grib_comment: Description of the param as it exists in the grib.

    Additional informational fields, not currently used in processing:
        grib_element: should be the short name of the param within the grib, but sometimes "unknown"
        grib_description:  description of the level, not the variable
    """

    grib_index_param: str
    grib_comment: str

    grib_index_level_type: Literal["sfc", "pl"] = "sfc"  # surface or pressure level
    grib_index_level_value: float = float("nan")

    # additional informational metadata, not currently used in processing:
    grib_element: str
    grib_description: str

    scale_factor: float | None = None

    # ECMWF will sometimes add variables to a dataset after the dataset start date.
    # This internal attribute can be used to handle whether or not we should try
    # to process a variable for a given date.
    date_available: Timestamp | None = None

    window_reset_frequency: Timedelta | None = pd.Timedelta.max
    deaccumulation_invalid_below_threshold_rate: float | None = None


class EcmwfDataVar(DataVar[EcmwfInternalAttrs]):
    pass


def vars_available(
    data_var_group: Sequence[EcmwfDataVar], init_time: Timestamp
) -> bool:
    """Check if a group of vars (which must share the same date_available) are available at init_time."""
    date_available = item({v.internal_attrs.date_available for v in data_var_group})
    return date_available is None or date_available <= init_time
