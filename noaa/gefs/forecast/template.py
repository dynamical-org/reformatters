from collections.abc import Hashable
from pathlib import Path
from typing import Literal

import dask
import dask.array
import numpy as np
import xarray as xr

from common.config import Config  # noqa:F401
from common.types import DatetimeLike, StoreLike

from .template_config import (
    CHUNKS,
    CHUNKS_ORDERED,
    COORDINATES,
    DATASET_ATTRIBUTES,
    DIMS,
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
    ds = ds.chunk(init_time=CHUNKS["init_time"])
    # Recompute valid time after reindex
    template_valid_time = ds["valid_time"]
    ds.coords["valid_time"] = ds["init_time"] + ds["lead_time"]
    ds["valid_time"].encoding = template_valid_time.encoding
    ds["valid_time"].attrs = template_valid_time.attrs

    # Coordinates which are dask arrays are not written with
    # to_zarr(store, compute=False) so we ensure all coordinates are loaded.
    for coordinate in ds.coords.values():
        assert isinstance(coordinate.data, np.ndarray)

    # Uncomment to make smaller zarr while developing
    if Config.is_dev():
        ds = (
            ds[["u100", "t2m"]]
            .isel(ensemble_member=slice(3))
            .sel(lead_time=["3h", "180h", "840h"])
        )

    return ds


def update_template() -> None:
    # Resolve to absolue path before changing directories
    template_path = Path(TEMPLATE_PATH).absolute()

    coords = get_template_dimension_coordinates()

    data_vars = {
        var_config.name: (
            DIMS,
            dask.array.full(  # type: ignore
                shape=tuple(len(coords[dim]) for dim in DIMS),
                fill_value=np.nan,
                dtype=var_config.encoding.dtype,
                chunks=CHUNKS_ORDERED,
            ),
        )
        for var_config in DATA_VARIABLES
    }

    ds = xr.Dataset(data_vars, coords, DATASET_ATTRIBUTES.model_dump())

    # This could be computed by users on the fly, but it compresses
    # really well so lets make things easy for users
    ds.coords["valid_time"] = ds["init_time"] + ds["lead_time"]

    for var_config in DATA_VARIABLES:
        data_var = ds[var_config.name]
        data_var.attrs = var_config.attrs.model_dump(exclude_none=True)
        data_var.encoding = var_config.encoding.model_dump(exclude_none=True)

    for coord_config in COORDINATES:
        ds.coords[coord_config.name].encoding = coord_config.encoding.model_dump(
            exclude_none=True
        )

    write_metadata(ds, template_path, mode="w")


def write_metadata(
    template_ds: xr.Dataset,
    store: StoreLike,
    mode: Literal["w", "w-", "a", "a-", "r+", "r"],
) -> None:
    template_ds.to_zarr(store, mode=mode, compute=False)
    print(f"Wrote metadata to {store} with mode {mode}.")


def chunk_args(ds: xr.Dataset) -> dict[Hashable, int]:
    """Returns {dim: chunk_size} mapping suitable to pass to ds.chunk()"""
    return {dim: chunk_sizes[0] for dim, chunk_sizes in ds.chunksizes.items()}
