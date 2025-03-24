from pathlib import Path
from typing import Any

import numpy as np
import rioxarray  # noqa: F401  Adds .rio accessor to datasets
import xarray as xr

from reformatters.common.template_utils import assign_var_metadata, make_empty_variable
from reformatters.common.template_utils import (
    write_metadata as write_metadata,  # re-export
)

from .template_config import (
    COORDINATES,
    DATA_VARIABLES,
    DATASET_ATTRIBUTES,
    DIMS,
    EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR,
    get_template_dimension_coordinates,
)
from .template_config import DATASET_ID as DATASET_ID

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "latest.zarr"


def update_template() -> None:
    coords = get_template_dimension_coordinates()

    data_vars = {
        var_config.name: make_empty_variable(DIMS, coords, var_config.encoding.dtype)
        for var_config in DATA_VARIABLES
    }

    ds = xr.Dataset(data_vars, coords, DATASET_ATTRIBUTES.model_dump(exclude_none=True))

    ds = ds.assign_coords(derive_coordinates(ds))

    ds = ds.rio.write_crs(
        "+proj=lcc +a=6371229 +b=6371229 +lon_0=262.5 +lat_0=38.5 +lat_1=38.5 +lat_2=38.5"
    )

    assert {d.name for d in DATA_VARIABLES} == set(ds.data_vars)
    for var_config in DATA_VARIABLES:
        assign_var_metadata(ds[var_config.name], var_config)

    assert {c.name for c in COORDINATES} == set(ds.coords)
    for coord_config in COORDINATES:
        # Don't overwrite -- retain the attributes that .rio.write_crs adds
        if coord_config.name == "spatial_ref":
            continue

        assign_var_metadata(ds.coords[coord_config.name], coord_config)

    write_metadata(ds, _TEMPLATE_PATH, mode="w")


def derive_coordinates(
    ds: xr.Dataset,
) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
    return {
        # TODO: Derive these coordinates from an actual model file
        "latitude": (("y", "x"), np.full((ds["y"].size, ds["x"].size), 0.0)),
        # TODO: Derive these coordinates from an actual model file
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
