from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from reformatters.common.template_utils import make_empty_variable
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    DATA_VARIABLES,
    DATASET_ATTRIBUTES,
    DIMS,
    EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR,
    get_template_dimension_coordinates,
)

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "latest.zarr"


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
    return {
        "latitude": (("y", "x"), np.full((ds["y"].size, ds["x"].size), 0.0)),
        "longitude": (("y", "x"), np.full((ds["y"].size, ds["x"].size), 0.0)),
        "ingested_forecast_length": (
            ("init_time",),
            np.full((ds["init_time"].size), np.timedelta64("NaT", "ns")),
        ),
        "expected_forecast_length": (
            ("init_time",),
            EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR.loc[ds["init_time"].dt.hour],
        ),
        "valid_time": ds["init_time"] + ds["lead_time"],
    }
