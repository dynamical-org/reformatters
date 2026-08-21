from typing import Literal

from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.types import Timedelta


class EcmwfIfsEns46DayInternalAttrs(BaseInternalAttrs):
    """Variable specific attributes used internally to drive processing. Not written to the dataset."""

    # The ECDS `variable` this data variable is retrieved as.
    ecds_variable: str
    # The GRIB element name GDAL reports, checked against every message read.
    grib_element: str
    # The GRIB level description GDAL reports, checked against every message read.
    grib_description: str
    # The bracketed unit of the GRIB comment GDAL reports, e.g. "[K]". Reads disable
    # GDAL's unit normalization, so this is the raw GRIB unit.
    grib_unit: str

    # Applied to the raw GRIB values in this order: value * scale_factor + add_offset.
    scale_factor: float | None = None
    add_offset: float | None = None

    deaccumulation_type: Literal["nonnegative", "signed"] = "nonnegative"
    deaccumulation_invalid_below_threshold_rate: float | None = None
    window_reset_frequency: Timedelta | None = None


class EcmwfIfsEns46DayDataVar(DataVar[EcmwfIfsEns46DayInternalAttrs]):
    pass
