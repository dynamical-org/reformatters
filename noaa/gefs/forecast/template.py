from collections.abc import Hashable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from common.config import Config  # noqa:F401
from common.download_directory import cd_into_download_directory
from common.types import DatetimeLike, StoreLike
from noaa.gefs.forecast.read_data import download_file

from .template_config import (
    CHUNKS,
    COORDINATES,
    DATA_VARIABLES,
    ENCODING,
    Dim,
    get_init_time_coordinates,
    get_template_coordinates,
)

TEMPLATE_PATH = "noaa/gefs/forecast/templates/latest.zarr"


def get_template(init_time_end: DatetimeLike) -> xr.Dataset:
    ds: xr.Dataset = xr.open_zarr(TEMPLATE_PATH)

    # Expand init_time dimension with complete coordinates
    ds = ds.reindex(init_time=get_init_time_coordinates(init_time_end))
    # Init time chunks are 1 when stored, set them to desired.
    ds = ds.chunk(init_time=CHUNKS["init_time"])
    # Recompute valid time after reindex
    ds.coords["valid_time"] = ds["init_time"] + ds["lead_time"]

    # Coordinates which are dask arrays are not written with
    # to_zarr(store, compute=False) so we ensure all coordinates are loaded.
    for coordinate in ds.coords.values():
        assert isinstance(coordinate.data, np.ndarray)

    # Uncomment to make smaller zarr while developing
    if Config.is_dev():
        ds = ds.isel(ensemble_member=slice(5), lead_time=slice(24))

    return ds


def update_template() -> None:
    coords = get_template_coordinates()

    # Resolve to absolue path before changing directories
    template_path = Path(TEMPLATE_PATH).absolute()

    # Pull a single file to load variable names and metadata.
    # Use a lead time > 0 because not all variables are present at lead time == 0.
    with cd_into_download_directory() as directory:
        path = download_file(
            pd.Timestamp("2024-01-01T00:00"),
            0,
            "s+a",
            pd.Timedelta("3h"),
            [_CUSTOM_ATTRIBUTES[var] for var in ["u10", "t2m"]],
            directory,
        )
        ds = read_file(path.name)

        # Expand ensemble and lead time dimensions + set coordinates and chunking
        ds = (
            ds.sel(ensemble_member=coords["ensemble_member"], method="nearest")
            .sel(lead_time=coords["lead_time"], method="nearest")
            .assign_coords(coords)
            .chunk(CHUNKS)
        )

        # Remove left over coordinates encoding for coords we don't keep
        for data_var in ds.data_vars.values():
            del data_var.encoding["coordinates"]

        # This could be computed by users on the fly, but it compresses
        # really well so lets make things easy for users
        ds.coords["valid_time"] = ds["init_time"] + ds["lead_time"]

        # TODO
        # correct temperature units from K -> C
        # Add custom attributes
        # Add add dataset wide attributes (dataset_id)
        # for var_name, data_var in ds.data_vars.items():
        #     data_var.attrs.update(_CUSTOM_ATTRIBUTES[var_name])

        write_metadata(ds, template_path, mode="w")


def write_metadata(
    template_ds: xr.Dataset,
    store: StoreLike,
    mode: Literal["w", "w-", "a", "a-", "r+", "r"],
) -> None:
    template_ds.to_zarr(store, mode=mode, compute=False, encoding=ENCODING)
    print(f"Wrote metadata to {store} with mode {mode}.")


def chunk_args(ds: xr.Dataset) -> dict[Hashable, int]:
    """Returns {dim: chunk_size} mapping suitable to pass to ds.chunk()"""
    return {dim: chunk_sizes[0] for dim, chunk_sizes in ds.chunksizes.items()}
