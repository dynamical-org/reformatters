import numpy as np

from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    mask_source_fill_value_inplace,
)
from reformatters.common.types import ArrayFloat32, Timedelta


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


class NoaaDataVar(DataVar[NoaaInternalAttrs]):
    pass


def mask_noaa_source_fill_values_inplace(
    values: ArrayFloat32, data_var: DataVar[NoaaInternalAttrs]
) -> None:
    mask_source_fill_value_inplace(values, data_var.internal_attrs)
    if data_var.name == "percent_frozen_precipitation_surface":
        values[values < 0] = np.nan
