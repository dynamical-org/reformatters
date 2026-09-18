"""Delete chunks, or whole arrays, from a dataset's stores.

No backfill operation removes anything: metadata is copied from the template into the
store rather than synced with it, and writes only ever set chunks. So an array the
template no longer declares, and chunks a variable should no longer carry, survive every
backfill. A deleted chunk reads as the array's fill value.
"""

import asyncio
import itertools
import math
from collections.abc import Sequence

import pandas as pd
import xarray as xr
import zarr
import zarr.core.metadata.v3
from zarr.abc.store import Store

from reformatters.common import template_utils
from reformatters.common.logging import get_logger
from reformatters.common.storage import StoreFactory, commit_if_icechunk
from reformatters.common.types import AppendDim

log = get_logger(__name__)

# Deletes are independent, so issue them concurrently; a batch bounds memory and gives
# the progress log something to report against.
_DELETE_BATCH = 2_000


def remove_from_stores(
    store_factory: StoreFactory,
    append_dim: AppendDim,
    arrays: Sequence[str],
    *,
    lead_index: int | None = None,
    before: pd.Timestamp | None = None,
    at: Sequence[pd.Timestamp] = (),
    whole_array: bool = False,
    apply: bool = False,
) -> None:
    """Delete the selected chunks of `arrays`, or the arrays themselves, from the
    primary store and every replica, committing icechunk stores and reading back
    what was deleted. Without `apply` only logs the plan."""
    chunk_selectors = lead_index is not None or before is not None or len(at) > 0
    assert chunk_selectors != whole_array, (
        "Give a chunk selector (--lead-index, --before, --at) or --whole-array, not both."
    )

    store = store_factory.primary_store()
    log.info(f"Store: {store_factory.primary_url()}")
    root = zarr.open_group(store, mode="r")
    positions = pd.DatetimeIndex(xr.open_zarr(store, chunks=None)[append_dim].values)

    plan: dict[str, list[str]] = {}
    for path in arrays:
        assert path in root, f"Not in the store, refusing to run: {path}"
        array = root[path]
        assert isinstance(array, zarr.Array), path
        keys = (
            []
            if whole_array
            else chunk_keys(
                path,
                array,
                append_dim,
                positions,
                lead_index=lead_index,
                before=before,
                at=at,
            )
        )
        plan[path] = keys
        what = "whole array" if whole_array else f"{len(keys)} chunk(s)"
        log.info(f"  {path}: shape {array.shape} shards {_shards(array)} -> {what}")

    total = sum(len(keys) for keys in plan.values())
    assert whole_array or total > 0, "Selector matched no chunks; nothing to do."
    summary = (
        f"{len(plan)} array(s): {', '.join(plan)}"
        if whole_array
        else f"{total} chunk(s) from {len(plan)} array(s): {', '.join(plan)}"
    )

    if not apply:
        log.info(f"Dry run. Would delete {summary}. Pass --apply.")
        return

    primary = store_factory.primary_store(writable=True)
    replicas = store_factory.replica_stores(writable=True)
    for writable in [primary, *replicas]:
        log.info(f"Deleting from {writable}")
        if whole_array:
            _delete_arrays(writable, list(plan))
        else:
            _delete_chunks(writable, plan, total)
    commit_if_icechunk(f"Delete {summary}", primary, replicas)
    log.info(f"Committed: Delete {summary}")

    for readonly in [store_factory.primary_store(), *store_factory.replica_stores()]:
        _verify_removed(readonly, plan, whole_array=whole_array)
    log.info("Verified removed from every store.")


def chunk_keys(
    path: str,
    array: zarr.Array,
    append_dim: AppendDim,
    positions: pd.DatetimeIndex,
    *,
    lead_index: int | None = None,
    before: pd.Timestamp | None = None,
    at: Sequence[pd.Timestamp] = (),
) -> list[str]:
    """Every store key of `array` the selectors match, in index order.

    A key holds a whole shard, so the selection must cover every position in each shard
    it touches. `at` positions must all be in the record; `before` and `at` may combine.
    """
    metadata = array.metadata
    assert isinstance(metadata, zarr.core.metadata.v3.ArrayV3Metadata), path
    dims = metadata.dimension_names
    assert dims is not None, path
    assert dims[0] == append_dim, (path, dims, append_dim)
    assert array.shape[0] == len(positions), (path, array.shape[0], len(positions))
    shards = _shards(array)
    per_axis = [
        list(range(math.ceil(size / shard)))
        for size, shard in zip(array.shape, shards, strict=True)
    ]

    selected: set[int] = set(
        range(len(positions) if before is None else int(positions.searchsorted(before)))
    )
    if at:
        wanted = positions.get_indexer(pd.DatetimeIndex(list(at)))
        missing = [t for t, i in zip(at, wanted, strict=True) if i < 0]
        assert not missing, f"{path}: positions not in the record: {missing}"
        selected &= set(wanted.tolist())
    per_axis[0] = _whole_shards(path, selected, shards[0], len(positions))

    if lead_index is not None:
        assert "lead_time" in dims, f"{path} has no lead_time axis to index"
        lead_axis = dims.index("lead_time")
        assert 0 <= lead_index < array.shape[lead_axis], (path, lead_index)
        assert shards[lead_axis] == 1, (
            f"{path}: a shard holds {shards[lead_axis]} leads, cannot delete one alone"
        )
        per_axis[lead_axis] = [lead_index]

    encode = metadata.chunk_key_encoding.encode_chunk_key
    return [f"{path}/{encode(index)}" for index in itertools.product(*per_axis)]


def _shards(array: zarr.Array) -> tuple[int, ...]:
    return array.shards or array.chunks


def _whole_shards(path: str, selected: set[int], shard: int, size: int) -> list[int]:
    """The shards along an axis that `selected` indices fill completely."""
    shards = sorted({i // shard for i in selected})
    partial = [
        s
        for s in shards
        if not set(range(s * shard, min((s + 1) * shard, size))) <= selected
    ]
    assert not partial, (
        f"{path}: shards {partial} hold {shard} positions each and the selection covers "
        "only some of them; a deleted shard takes every position in it"
    )
    return shards


def _delete_arrays(store: Store, paths: list[str]) -> None:
    root = zarr.open_group(store, mode="a")
    consolidated = root.metadata.consolidated_metadata is not None
    for path in paths:
        del root[path]
        log.info(f"  deleted array {path}")
    # Deleting a member leaves the consolidated metadata still listing it.
    if consolidated:
        with template_utils.ignore_consolidated_metadata_spec_warning():
            zarr.consolidate_metadata(store)


def _delete_chunks(store: Store, plan: dict[str, list[str]], total: int) -> None:
    done = 0
    for keys in plan.values():
        for batch in itertools.batched(keys, _DELETE_BATCH, strict=False):
            asyncio.run(_delete_all(store, batch))
            done += len(batch)
            log.info(f"  deleted {done}/{total}")


async def _delete_all(store: Store, keys: Sequence[str]) -> None:
    await asyncio.gather(*(store.delete(key) for key in keys))


def _verify_removed(
    store: Store, plan: dict[str, list[str]], *, whole_array: bool
) -> None:
    root = zarr.open_group(store, mode="r")
    for path, keys in plan.items():
        if whole_array:
            assert path not in root, f"{path} still present in {store}"
            continue
        for batch in itertools.batched(keys, _DELETE_BATCH, strict=False):
            present = asyncio.run(_existing(store, batch))
            assert not present, f"{path}: still present in {store}: {present[:5]}"


async def _existing(store: Store, keys: Sequence[str]) -> list[str]:
    exists = await asyncio.gather(*(store.exists(key) for key in keys))
    return [key for key, found in zip(keys, exists, strict=True) if found]
