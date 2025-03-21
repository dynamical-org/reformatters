from typing import Any

import numpy as np
import xarray as xr

from reformatters.common.template_utils import make_empty_variable
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    DATA_VARIABLES,
    DATASET_ATTRIBUTES,
    DIMS,
    get_template_dimension_coordinates,
)


def update_template() -> None:
    coords = get_template_dimension_coordinates()

    data_vars = {
        var_config.name: make_empty_variable(DIMS, coords, var_config.encoding.dtype)
        for var_config in DATA_VARIABLES
    }

    ds = xr.Dataset(data_vars, coords, DATASET_ATTRIBUTES.model_dump(exclude_none=True))

    ds.assign_coords(derive_coordinates(ds))


def derive_coordinates(
    ds: xr.Dataset,
) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
    return {}
