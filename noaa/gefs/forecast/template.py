from collections.abc import Hashable
from pathlib import Path
from typing import Any, Literal

import dask
import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from common.config import Config  # noqa:F401
from common.types import DatetimeLike, StoreLike

from .config_models import Coordinate, DataVar
from .template_config import (
    COORDINATES,
    DATASET_ATTRIBUTES,
    ENSEMBLE_VAR_CHUNKS,
    ENSEMBLE_VAR_CHUNKS_ORDERED,
    ENSEMBLE_VAR_DIMS,
    STATISTIC_VAR_CHUNKS,
    STATISTIC_VAR_CHUNKS_ORDERED,
    STATISTIC_VAR_DIMS,
    Dim,
    get_init_time_coordinates,
    get_template_dimension_coordinates,
)
from .template_config import (
    DATA_VARIABLES as DATA_VARIABLES,
)

TEMPLATE_PATH = "noaa/gefs/forecast/templates/latest.zarr"


def get_template(init_time_end: DatetimeLike) -> xr.Dataset:
    ds: xr.Dataset = xr.open_zarr(TEMPLATE_PATH)

    # Expand init_time dimension with complete coordinates
    ds = ds.reindex(init_time=get_init_time_coordinates(init_time_end))
    # Init time chunks are 1 when stored, set them to desired.
    assert ENSEMBLE_VAR_CHUNKS["init_time"] == STATISTIC_VAR_CHUNKS["init_time"]
    ds = ds.chunk(init_time=ENSEMBLE_VAR_CHUNKS["init_time"])
    # Recompute derived coordinates after reindex
    ds = add_derived_coordinates(ds)

    # Coordinates which are dask arrays are not written with .to_zarr(store, compute=False)
    # We want to write all coords when writing metadata, so ensure they are loaded as numpy arrays.
    for coordinate in ds.coords.values():
        coordinate.load()

    # Uncomment to make smaller dataset while developing
    # if Config.is_dev():
    #     ds = ds[
    #         [
    #             "wind_u_100m",
    #             "wind_u_10m",
    #             "wind_u_10m_avg",
    #             "temperature_2m",
    #             "temperature_2m_avg",
    #             "precipitation_surface",
    #             "precipitation_surface_avg",
    #             "percent_frozen_precipitation_surface",
    #         ]
    #     ].sel(
    #         ensemble_member=slice(3),
    #         lead_time=["0h", "3h", "90h", "240h", "840h"],
    #     )

    return ds


def update_template() -> None:
    # Resolve to absolue path before changing directories
    template_path = Path(TEMPLATE_PATH).absolute()

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

    assert {d.name for d in DATA_VARIABLES} == set(ds.data_vars)
    for var_config in DATA_VARIABLES:
        assign_var_metadata(ds[var_config.name], var_config)

    assert {c.name for c in COORDINATES} == set(ds.coords)
    for coord_config in COORDINATES:
        assign_var_metadata(ds.coords[coord_config.name], coord_config)

    write_metadata(ds, template_path, mode="w")


def add_derived_coordinates(ds: xr.Dataset, copy_metadata: bool = True) -> xr.Dataset:
    new_coords = {
        "ingested_forecast_length": (
            "init_time",
            np.full(ds["init_time"].size, np.timedelta64("NaT", "ns")),
        ),
        "expected_forecast_length": (
            "init_time",
            np.where(
                ds["init_time"].dt.hour == 0,
                pd.Timedelta(hours=840),
                pd.Timedelta(hours=384),
            ),  # type: ignore
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
    var: xr.DataArray, var_config: DataVar | Coordinate
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
    var_config: DataVar, coords: dict[Dim, Any]
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


def write_metadata(
    template_ds: xr.Dataset,
    store: StoreLike,
    mode: Literal["w", "w-", "a", "a-", "r+", "r"],
) -> None:
    template_ds.to_zarr(store, mode=mode, compute=False)
    print(f"Wrote metadata to {store} with mode {mode}.")


def chunk_args(da: xr.DataArray) -> dict[Hashable, int]:
    """Returns {dim: chunk_size} mapping suitable to pass to ds.chunk()"""
    return {dim: chunk_sizes[0] for dim, chunk_sizes in da.chunksizes.items()}
