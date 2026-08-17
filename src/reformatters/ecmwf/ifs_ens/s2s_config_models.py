from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.types import Timedelta


class EcmwfS2sInternalAttrs(BaseInternalAttrs):
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

    # Clamp packing artifacts that push values outside the variable's physical range.
    minimum_value: float | None = None
    maximum_value: float | None = None

    # None on a deaccumulated variable means its accumulation is signed, so its steps
    # are differenced without a validity threshold or a clamp to zero.
    deaccumulation_invalid_below_threshold_rate: float | None = None
    window_reset_frequency: Timedelta | None = None


class EcmwfS2sDataVar(DataVar[EcmwfS2sInternalAttrs]):
    pass
