from collections.abc import Sequence
from typing import Any

import numpy as np

from reformatters.common.config_models import BaseInternalAttrs, DataVar
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
    source_missing_value: float | None = None
    source_missing_value_atol: float = 0.0


class NoaaDataVar(DataVar[NoaaInternalAttrs]):
    pass


def source_missing_value_var_names(
    data_vars: Sequence[DataVar[Any]],
) -> tuple[str, ...]:
    return tuple(
        data_var.name
        for data_var in data_vars
        if getattr(data_var.internal_attrs, "source_missing_value", None) is not None
    )


def mask_source_missing_values_inplace(
    values: ArrayFloat32, internal_attrs: NoaaInternalAttrs
) -> None:
    missing_value = internal_attrs.source_missing_value
    if missing_value is None:
        return
    values[
        np.isclose(
            values,
            missing_value,
            rtol=0,
            atol=internal_attrs.source_missing_value_atol,
        )
    ] = np.nan
