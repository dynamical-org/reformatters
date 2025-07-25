from pathlib import Path

import rioxarray  # noqa: F401  Adds .rio accessor to datasets
import xarray as xr

from reformatters.common.config import Config
from reformatters.common.logging import get_logger
from reformatters.common.template_utils import (
    assign_var_metadata,
    empty_copy_with_reindex,
    make_empty_variable,
)
from reformatters.common.template_utils import (
    write_metadata as write_metadata,  # re-export
)
from reformatters.common.types import DatetimeLike

from .template_config import ANALYSIS_ENSEMBLE_MEMBER as ANALYSIS_ENSEMBLE_MEMBER

# Explicitly re-export the DATA_VARIABLES, DATASET_ID, DATASET_VERSION, and APPEND_DIMENSION
# which are part of the template module's public interface along with
# get_template and update_template.
from .template_config import APPEND_DIMENSION as APPEND_DIMENSION
from .template_config import (
    COORDINATES,
    DATASET_ATTRIBUTES,
    DIMS,
    get_template_dimension_coordinates,
    get_time_coordinates,
)
from .template_config import DATA_VARIABLES as DATA_VARIABLES
from .template_config import DATASET_ID as DATASET_ID
from .template_config import DATASET_VERSION as DATASET_VERSION

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "latest.zarr"

logger = get_logger(__name__)


def get_template(init_time_end: DatetimeLike) -> xr.Dataset:
    ds: xr.Dataset = xr.open_zarr(_TEMPLATE_PATH, decode_timedelta=True)

    # Expand init_time dimension with complete coordinates
    ds = empty_copy_with_reindex(
        ds, APPEND_DIMENSION, get_time_coordinates(init_time_end)
    )

    # Coordinates which are dask arrays are not written with .to_zarr(store, compute=False)
    # We want to write all coords when writing metadata, so ensure they are loaded as numpy arrays.
    for coordinate in ds.coords.values():
        coordinate.load()

    if not Config.is_prod:
        ds = ds[["precipitation_surface", "temperature_2m", "wind_u_100m"]]

    return ds


def update_template() -> None:
    coords = get_template_dimension_coordinates()

    data_vars = {
        var_config.name: make_empty_variable(DIMS, coords, var_config.encoding.dtype)
        for var_config in DATA_VARIABLES
    }

    ds = xr.Dataset(data_vars, coords, DATASET_ATTRIBUTES.model_dump(exclude_none=True))

    # Add coordinate reference system for out of the box rioxarray support
    ds = ds.rio.write_crs("+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs")
    ds["spatial_ref"].attrs["comment"] = (
        "This coordinate reference system matches the source data which "
        "follows WMO conventions of assuming the earth is a perfect sphere "
        "with a radius of 6,371,229m. It is similar to EPSG:4326, but "
        "EPSG:4326 uses a more accurate representation of the earth's shape."
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
