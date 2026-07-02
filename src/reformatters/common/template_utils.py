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
import zarr.abc.store
import zarr.storage
from zarr.abc.store import Store

from reformatters.common.config import Config
from reformatters.common.config_models import Coordinate, DataVar
from reformatters.common.iterating import walk_data_arrays
from reformatters.common.logging import get_logger
from reformatters.common.storage import StoreFactory, commit_if_icechunk
from reformatters.common.zarr import assert_fill_values_set

log = get_logger(__name__)


def _to_zarr_metadata(
    template: xr.DataTree, store: Store | Path, mode: Literal["w", "w-"]
) -> None:
    # safe_chunks=False avoids a per-chunk dask graph that OOMs large arrays (compute=False
    # writes no data chunks anyway). write_inherited_coords=True duplicates shared coords
    # into each group so groups open standalone.
    template.to_zarr(
        store,
        mode=mode,
        compute=False,
        write_inherited_coords=True,
        safe_chunks=False,
        write_empty_chunks=True,
    )


def write_metadata(
    template_ds: xr.DataTree,
    storage: zarr.storage.StoreLike | StoreFactory,
    mode: Literal["w", "w-"] | None = None,
    skip_icechunk_commit: bool = False,
) -> None:
    store: Store | Path
    replica_stores: list[Store]

    assert_fill_values_set(template_ds)

    if isinstance(storage, StoreFactory):
        store = storage.primary_store(writable=True)
        assert mode is None, "mode should not be provided if StoreFactory is provided"
        mode = storage.mode()
        replica_stores = storage.replica_stores(writable=True)
    else:
        assert isinstance(storage, (Store, Path))
        store = storage
        replica_stores = []
        if mode is None:
            assert isinstance(store, Path), f"Expected Path, got {type(store)}"
            mode = _get_mode_from_path_store(store)

    if mode == "w" and not isinstance(store, Path):
        raise ValueError(
            f"mode='w' is not allowed on remote stores (got {type(store).__name__}). "
            "Use copy_zarr_metadata to update metadata on existing stores."
        )

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
            _to_zarr_metadata(template_ds, replica_store, mode)

        log.info(f"Writing metadata to store {store} with mode {mode}")
        _to_zarr_metadata(template_ds, store, mode)

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

    if path_str.endswith(("templates/latest.zarr", "-tmp.zarr")):
        return "w"  # Allow overwritting template config latest.zarr and local_tmp_store

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


def _structural_signature(
    var: xr.DataArray, append_dim: str, *, compare_append_axis: bool
) -> dict[str, Any]:
    """Structural fields of a variable that must stay stable for an existing store.

    Compares the on-disk format (encoding dtype/chunks/shards) plus dims. dtype is
    normalized through np.dtype because an in-code template stores it as a string
    ("float32") while a reopened store reports a numpy dtype.

    When ``compare_append_axis`` is False the append dimension's chunk/shard size is
    dropped from the comparison (see assert_no_structural_drift_from_existing_store).
    """

    def chunk_sizes(
        value: tuple[int, ...] | list[int] | int | None,
    ) -> tuple[int, ...] | None:
        if value is None:
            return None
        sizes = (value,) if isinstance(value, int) else tuple(value)
        if compare_append_axis:
            return sizes
        return tuple(
            size for axis, size in enumerate(sizes) if var.dims[axis] != append_dim
        )

    suffix = "" if compare_append_axis else " (non-append)"
    return {
        "dims": tuple(var.dims),
        "dtype": np.dtype(var.encoding["dtype"]),
        f"chunks{suffix}": chunk_sizes(var.encoding.get("chunks")),
        f"shards{suffix}": chunk_sizes(var.encoding.get("shards")),
    }


def _non_append_dimension_coords(
    tree: xr.DataTree, append_dim: str
) -> dict[str, xr.DataArray]:
    """Dimension coordinates (across all groups) other than the append dim, by name —
    the 1D axis labels (latitude/longitude, pressure_level, model_level, …). Reordering
    one relabels existing chunks. Excludes the growing append dim, 2D auxiliary coords
    (projected lat/lon, valid_time), and scalars (spatial_ref), which need not match."""
    coords: dict[str, xr.DataArray] = {}
    for node in tree.subtree:
        for name, coord in node.to_dataset().coords.items():
            if coord.dims == (name,) and name != append_dim:
                coords[str(name)] = coord
    return coords


def assert_no_structural_drift_from_existing_store(
    template_ds: xr.DataTree, existing_ds: xr.DataTree, append_dim: str
) -> None:
    """Fail an operational update whose template would change the structure of the
    already-published store.

    Operational updates write new chunks into an existing archive and then swap in the
    template's metadata. If the template's structure has drifted from what readers
    currently see — a removed/renamed variable, or a changed dtype, dims, chunks, or
    shards — the update would corrupt the archive or break readers. This guard runs on
    worker 0 before any data is written and raises before damage is done.

    Compares every variable across all groups (by var.path) and every non-append
    dimension coordinate's values (a reordered vertical level or moved lat/lon would
    relabel existing chunks). Only variables/coords present in `existing_ds` are checked, so the
    template may freely add new ones. Backfills are exempt (they rewrite the whole store
    and may legitimately change structure), so callers only run this for operational updates.

    The append dimension's chunk/shard size is compared in dev/prod but skipped under
    Config.env == test, where the integration auto-shrink helper sizes append-dim chunks
    to the (varying) template length. In production these sizes are fixed config, so any
    change is caught here as well as at PR time by the template-drift test.
    """
    compare_append_axis = not Config.is_test
    template_by_path = dict(walk_data_arrays(template_ds))
    mismatches: list[str] = []
    for path, existing_var in walk_data_arrays(existing_ds):
        if path not in template_by_path:
            mismatches.append(
                f"{path}: in existing store but missing from update template"
            )
            continue

        existing_sig = _structural_signature(
            existing_var, append_dim, compare_append_axis=compare_append_axis
        )
        template_sig = _structural_signature(
            template_by_path[path], append_dim, compare_append_axis=compare_append_axis
        )
        for field, existing_value in existing_sig.items():
            template_value = template_sig[field]
            if existing_value != template_value:
                mismatches.append(
                    f"{path}.{field}: existing store has {existing_value!r}, "
                    f"update template has {template_value!r}"
                )

    template_coords = _non_append_dimension_coords(template_ds, append_dim)
    for name, existing_coord in _non_append_dimension_coords(
        existing_ds, append_dim
    ).items():
        if name not in template_coords:
            mismatches.append(
                f"coord {name}: in existing store but missing from update template"
            )
        elif not existing_coord.equals(template_coords[name]):
            mismatches.append(f"coord {name}: values differ from the existing store")

    if mismatches:
        raise ValueError(
            "Update template structure does not match the existing store. "
            "Operational updates cannot change the structure of a published dataset; "
            "run a backfill to change structure. Mismatches:\n"
            + "\n".join(f"- {m}" for m in mismatches)
        )


def make_empty_variable(
    dims: tuple[str, ...],
    coords: dict[str, Any],
    dtype: np.typing.DTypeLike,
) -> xr.Variable:
    shape = tuple(len(coords[dim]) for dim in dims)

    array = dask.array.full(
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
        nan_array = dask.array.full(
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
