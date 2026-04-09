from collections.abc import Sequence
from typing import Literal

import pandas as pd

from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.iterating import item
from reformatters.common.pydantic import FrozenBaseModel, replace
from reformatters.common.types import Timedelta, Timestamp


class MarsSourceOverrides(FrozenBaseModel):
    """Overrides for variables where the MARS archive stores data differently than open data.

    Only set fields that differ. For example, MARS stores geopotential (z, m²/s²)
    rather than geopotential height (gh, gpm), requiring overrides for param, comment, and a
    scale factor to convert.
    """

    grib_index_param: str | None = None
    grib_element: str | None = None
    grib_comment: str | None = None
    scale_factor: float | None = None


class EcmwfInternalAttrs(BaseInternalAttrs):
    """Variable specific attributes used internally to drive processing. Not written to the dataset."""

    # The short name of the param as it exists in the index file. Does not map to any names in the grib.
    grib_index_param: str
    # ECMWF sometimes uses different param names in the index at certain lead time ranges.
    # Each tuple is (start_lead_time, end_lead_time, override_param).
    grib_index_param_lead_time_overrides: tuple[
        tuple[Timedelta, Timedelta, str], ...
    ] = ()
    # Description of the param as it exists in the grib.
    grib_comment: str

    grib_index_level_type: Literal["sfc", "pl"] = "sfc"  # surface or pressure level
    grib_index_level_value: float = float("nan")

    # Grib attributes used to select correct message
    grib_element: str  # short name of the param within the grib, sometimes "unknown"
    grib_description: str  # description of the level, not the variable

    scale_factor: float | None = None

    # Date when this variable became available for the time period being processed.
    # Source-specific overrides (e.g. _resolve_mars_data_var) may clear this when
    # the source has the variable regardless of the original availability date.
    date_available: Timestamp | None = None

    window_reset_frequency: Timedelta | None = pd.Timedelta.max
    deaccumulation_invalid_below_threshold_rate: float | None = None

    mars: MarsSourceOverrides | None = None


class EcmwfDataVar(DataVar[EcmwfInternalAttrs]):
    pass


def vars_available(
    data_var_group: Sequence[EcmwfDataVar], init_time: Timestamp
) -> bool:
    """Check if a group of vars (which must share the same date_available) are available at init_time."""
    date_available = item({v.internal_attrs.date_available for v in data_var_group})
    return date_available is None or date_available <= init_time


def _resolve_grib_index_param(
    data_var: EcmwfDataVar, lead_time: Timedelta
) -> EcmwfDataVar:
    for (
        start,
        end,
        param_override,
    ) in data_var.internal_attrs.grib_index_param_lead_time_overrides:
        if start <= lead_time <= end:
            return replace(
                data_var,
                internal_attrs=replace(
                    data_var.internal_attrs, grib_index_param=param_override
                ),
            )
    return data_var


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
