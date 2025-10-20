import pandas as pd

from reformatters.common.config_models import BaseInternalAttrs, DataVar
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

    # additional informational metadata, not currently used in processing:
    grib_element: str
    grib_description: str

    scaling_factor: float | None = None

    # ECMWF will sometimes add variables to a dataset after the dataset start date.
    # This internal attribute can be used to handle whether or not we should try
    # to process a variable for a given date.
    date_available: Timestamp | None = None

    window_reset_frequency: Timedelta | None = pd.Timedelta.max
    deaccumulation_invalid_below_threshold_rate: float | None = None


class EcmwfDataVar(DataVar[EcmwfInternalAttrs]):
    pass
