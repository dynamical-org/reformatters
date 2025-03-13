import json
import warnings
from collections.abc import Callable, Sized
from pathlib import Path
from typing import Any, Literal

import dask.array
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from reformatters.common.config_models import Coordinate, DataVar
from reformatters.common.logging import get_logger

logger = get_logger(__name__)


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

    if isinstance(store, Path | str):
        sort_consolidated_metadata(Path(store) / "zarr.json")


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


def make_empty_variable(
    dims: tuple[str, ...],
    coords: dict[str, Sized | np.ndarray[Any, Any] | pd.Index[Any]],
    dtype: np.typing.DTypeLike,
) -> xr.Variable:
    shape = tuple(len(coords[dim]) for dim in dims)

    array = dask.array.full(  # type: ignore
        shape=shape,
        fill_value=np.nan,
        dtype=dtype,
        chunks=-1,  # see encoding's chunk/shards for stored chunks
    )

    return xr.Variable(dims, array)


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


def empty_copy_with_reindex(
    template_ds: xr.Dataset,
    dim: str,
    new_coords: Sized,
    derive_coordinates_fn: Callable[[xr.Dataset], dict[str, xr.DataArray]]
    | None = None,
) -> xr.Dataset:
    # Skip coords where one of the dims is `dim`, those need to
    # either be provided as `new_coords` or derived afterwards.
    coords = {v: c for v, c in template_ds.coords.items() if dim not in c.dims}
    coords = {dim: new_coords, **coords}

    ds = xr.Dataset({}, coords, template_ds.attrs.copy())

    if derive_coordinates_fn is not None:
        ds = ds.assign_coords(derive_coordinates_fn(ds))

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
