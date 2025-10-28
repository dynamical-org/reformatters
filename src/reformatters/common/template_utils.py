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
from reformatters.common.storage import StoreFactory, commit_if_icechunk
from reformatters.common.zarr import assert_fill_values_set

log = get_logger(__name__)


def write_metadata(
    template_ds: xr.Dataset,
    storage: zarr.storage.StoreLike | StoreFactory,
    mode: Literal["w", "w-"] | None = None,
    skip_icechunk_commit: bool = False,
) -> None:
    store: zarr.abc.store.Store | Path
    replica_stores: list[zarr.abc.store.Store]

    assert_fill_values_set(template_ds)

    if isinstance(storage, StoreFactory):
        store = storage.primary_store(writable=True)
        assert mode is None, "mode should not be provided if StoreFactory is provided"
        mode = storage.mode()
        replica_stores = storage.replica_stores(writable=True)
    else:
        assert isinstance(storage, (zarr.abc.store.Store, Path))
        store = storage
        replica_stores = []
        # respect mode if provided by legacy implementations
        if mode is None:
            assert isinstance(store, Path), f"Expected Path, got {type(store)}"
            mode = _get_mode_from_path_store(store)

    with warnings.catch_warnings():
        # Unconsolidated metadata is also written so adding
        # consolidated metadata is unlikely to impact interoperability.
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part in the Zarr format 3 specification",
            category=UserWarning,
        )

        for replica_store in replica_stores:
            log.info(f"Writing metadata to replica {replica_store} with mode {mode}")
            template_ds.to_zarr(replica_store, mode=mode, compute=False)  # type: ignore[call-overload]

        log.info(f"Writing metadata to store {store} with mode {mode}")
        template_ds.to_zarr(store, mode=mode, compute=False)  # type: ignore[call-overload]

    if isinstance(store, Path | str):
        sort_consolidated_metadata(Path(store) / "zarr.json")

    if not skip_icechunk_commit:
        commit_if_icechunk(
            message=f"Metadata written at {pd.Timestamp.now(tz='UTC').isoformat()}",
            primary_store=store,
            replica_stores=replica_stores,
        )


def _get_mode_from_path_store(store: Path) -> Literal["w", "w-"]:
    if store.parent.name == "templates":
        path_str = f"templates/{store.name}"
    else:
        path_str = store.name

    if path_str.endswith(("templates/latest.zarr", "dev.zarr", "-tmp.zarr")):
        return "w"  # Allow overwritting dev store and template config latest.zarr

    return "w-"  # Safe default - don't overwrite


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
    coords: dict[str, Any],
    dtype: np.typing.DTypeLike,
) -> xr.Variable:
    shape = tuple(len(coords[dim]) for dim in dims)

    array = dask.array.full(  # type: ignore[no-untyped-call]
        shape=shape,
        fill_value=np.nan,
        dtype=dtype,
        chunks=-1,
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
    derive_coordinates_fn: Callable[
        [xr.Dataset],
        dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]],
    ]
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
        nan_array = dask.array.full(  # type:ignore[no-untyped-call]
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
