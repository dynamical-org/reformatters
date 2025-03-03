import json
import warnings
from collections.abc import Sized
from pathlib import Path
from typing import Any, Literal

import dask
import dask.array
import numpy as np
import rioxarray  # noqa: F401  Adds .rio accessor to datasets
import xarray as xr
import zarr

from reformatters.common.config import Config  # noqa:F401
from reformatters.common.config_models import Coordinate, DataVar
from reformatters.common.logging import get_logger
from reformatters.common.types import DatetimeLike

from .template_config import (
    COORDINATES,
    DATASET_ATTRIBUTES,
    ENSEMBLE_VAR_CHUNKS_ORDERED,
    ENSEMBLE_VAR_DIMS,
    EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR,
    STATISTIC_VAR_CHUNKS_ORDERED,
    STATISTIC_VAR_DIMS,
    Dim,
    get_init_time_coordinates,
    get_template_dimension_coordinates,
)

# Explicitly re-export the DATA_VARIABLES, DATASET_ID, and DATASET_VERSION
# which are part of the template module's public interface along with
# get_template and update_template.
from .template_config import (
    DATA_VARIABLES as DATA_VARIABLES,
)
from .template_config import (
    DATASET_ID as DATASET_ID,
)
from .template_config import (
    DATASET_VERSION as DATASET_VERSION,
)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "latest.zarr"

logger = get_logger(__name__)


def get_template(init_time_end: DatetimeLike) -> xr.Dataset:
    ds: xr.Dataset = xr.open_zarr(TEMPLATE_PATH, decode_timedelta=True)

    # Expand init_time dimension with complete coordinates
    ds = empty_copy_with_reindex(
        ds, "init_time", get_init_time_coordinates(init_time_end)
    )

    # Coordinates which are dask arrays are not written with .to_zarr(store, compute=False)
    # We want to write all coords when writing metadata, so ensure they are loaded as numpy arrays.
    for coordinate in ds.coords.values():
        coordinate.load()

    # Uncomment to make smaller dataset while developing
    # if Config.is_dev:
    #     ds = ds[
    #         [
    #             "wind_u_10m",
    #             "wind_u_10m_avg",
    #             "wind_u_100m",
    #             "temperature_2m",
    #             "temperature_2m_avg",
    #             "precipitation_surface",
    #             "precipitation_surface_avg",
    #             "percent_frozen_precipitation_surface",
    #         ]
    #     ].sel(
    #         ensemble_member=slice(2),
    #         lead_time=[
    #             "0h",
    #             "3h",
    #             "240h",
    #             "840h",
    #         ],
    #     )

    return ds


def update_template() -> None:
    coords = get_template_dimension_coordinates()

    data_vars = {
        var_config.name: construct_data_variable(var_config, coords)
        for var_config in DATA_VARIABLES
    }

    ds = xr.Dataset(data_vars, coords, DATASET_ATTRIBUTES.model_dump(exclude_none=True))

    # Skip copying metadata (encoding and attributes) because the
    # coordinates don't already exist on ds. Encoding and attributes
    # will be added from the COORDINATES configs below.
    ds = add_derived_coordinates(ds, copy_metadata=False)

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

    write_metadata(ds, TEMPLATE_PATH, mode="w")


def add_derived_coordinates(ds: xr.Dataset, copy_metadata: bool = True) -> xr.Dataset:
    new_coords = {
        "ingested_forecast_length": (
            ["init_time", "ensemble_member"],
            np.full(
                (ds["init_time"].size, ds["ensemble_member"].size),
                np.timedelta64("NaT", "ns"),
            ),
        ),
        "expected_forecast_length": (
            "init_time",
            EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR.loc[ds["init_time"].dt.hour],
        ),
        "valid_time": ds["init_time"] + ds["lead_time"],
    }

    new_ds = ds.assign_coords(new_coords)

    if copy_metadata:
        for coord_name in new_coords.keys():
            new_ds[coord_name].attrs = ds[coord_name].attrs
            new_ds[coord_name].encoding = ds[coord_name].encoding

    return new_ds


def assign_var_metadata(
    var: xr.DataArray, var_config: DataVar[Any] | Coordinate
) -> xr.DataArray:
    var.encoding = var_config.encoding.model_dump(exclude_none=True)

    # Encoding time data requires a `units` key in `encoding`.
    # Ensure the value matches units value in the usual `attributes` location.
    if var_config.encoding.units is not None and var_config.attrs.units is not None:
        assert var_config.encoding.units == var_config.attrs.units

    var.attrs = {
        k: v
        for k, v in var_config.attrs.model_dump(exclude_none=True).items()
        if k != "units" or "units" not in var.encoding
    }
    return var


def construct_data_variable(
    var_config: DataVar[Any], coords: dict[Dim, Any]
) -> tuple[tuple[Dim, ...], dask.array.Array]:  # type: ignore[name-defined]
    if var_config.attrs.ensemble_statistic is None:
        dims = ENSEMBLE_VAR_DIMS
        chunks = ENSEMBLE_VAR_CHUNKS_ORDERED
    else:
        dims = STATISTIC_VAR_DIMS
        chunks = STATISTIC_VAR_CHUNKS_ORDERED

    shape = tuple(len(coords[dim]) for dim in dims)

    array = dask.array.full(  # type: ignore
        shape=shape,
        fill_value=np.nan,
        dtype=var_config.encoding.dtype,
        chunks=chunks,
    )

    return dims, array


def empty_copy_with_reindex(
    template_ds: xr.Dataset, dim: Dim, new_coords: Sized
) -> xr.Dataset:
    # Skip coords where one of the dims is `dim`, those need to
    # either be provided as `new_coords` or derived afterwards.
    coords = {v: c for v, c in template_ds.coords.items() if dim not in c.dims}
    coords = {dim: new_coords, **coords}

    ds = xr.Dataset({}, coords, template_ds.attrs.copy())

    ds = add_derived_coordinates(ds, copy_metadata=False)

    for coord_name, template_coord in template_ds.coords.items():
        ds[coord_name].attrs = template_coord.attrs.copy()
        ds[coord_name].encoding = template_coord.encoding.copy()

    for var_name, var in template_ds.data_vars.items():
        nan_array = dask.array.full(  # type:ignore[no-untyped-call,attr-defined]
            fill_value=np.nan,
            shape=[ds.sizes[dim] for dim in var.dims],
            dtype=var.dtype,
            # Using actual chunk size causes OOM when writing metadata.
            # Check the chunks/shards in the encoding for the stored sizes.
            chunks=-1,
        )
        ds[var_name] = (var.dims, nan_array)
        ds[var_name].attrs = var.attrs.copy()
        ds[var_name].encoding = var.encoding.copy()

    return ds


def write_metadata(
    template_ds: xr.Dataset,
    store: zarr.storage.StoreLike,
    mode: Literal["w", "w-"],
) -> None:
    with warnings.catch_warnings():
        # Unconsolidated metadata is also written so adding
        # consolidated metadata is unlikely to impact interoperability.
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part in the Zarr format 3 specification",
            category=UserWarning,
        )
        template_ds.to_zarr(store, mode=mode, compute=False)  # type: ignore[call-overload]
    logger.info(f"Wrote metadata to {store} with mode {mode}.")

    if isinstance(store, Path):
        sort_consolidated_metadata(store / "zarr.json")


def sort_consolidated_metadata(zarr_json_path: Path) -> None:
    """
    Sort the variable and coordinates in the consolidated metadata
    so template diffs are easier to read.
    """
    with open(zarr_json_path) as f:
        zarr_json = json.load(f)

    zarr_json["consolidated_metadata"]["metadata"] = dict(
        sorted(zarr_json["consolidated_metadata"]["metadata"].items())
    )

    with open(zarr_json_path, "w") as f:
        json.dump(zarr_json, f, indent=2)
