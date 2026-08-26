"""Delete chunks, or whole arrays, from a dataset's icechunk store.

A backfill never removes anything: metadata is copied from the template into the store
rather than synced with it, and the virtual write path only ever sets refs. So chunks a
variable should no longer carry survive every backfill. Deleting a chunk leaves the
array intact and makes that position read as its fill value.

Reads back what it deleted before committing, and is a dry run unless given --apply.

    # A field HRRR writes as a constant at forecast hour 0: drop lead 0, keep the rest.
    DYNAMICAL_ENV=prod uv run src/scripts/remove_refs.py noaa-hrrr-forecast-48-hour-virtual \
        --array precipitation_rate_surface --lead-index 0

    # A field unusable before a source version boundary: drop the positions before it.
    DYNAMICAL_ENV=prod uv run src/scripts/remove_refs.py noaa-hrrr-analysis-virtual \
        --array model_level/turbulent_kinetic_energy --before 2018-07-12T12:00

    # No chunk selector deletes the whole array.
    DYNAMICAL_ENV=prod uv run src/scripts/remove_refs.py <dataset-id> --array <name>
"""

import argparse
import asyncio
import math
import sys
from collections.abc import Iterator
from typing import Any

import icechunk
import pandas as pd
import xarray as xr
import zarr
from zarr.abc.store import Store

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.logging import get_logger
from reformatters.common.storage import StoreFactory

log = get_logger(__name__)

# Deletes are independent, so issue them concurrently; a batch bounds memory and gives
# the progress log something to report against.
_DELETE_BATCH = 2_000


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_id")
    parser.add_argument("--array", action="append", required=True, dest="arrays")
    parser.add_argument(
        "--lead-index",
        type=int,
        help="Delete only this lead_time index, at every append-dim position.",
    )
    parser.add_argument(
        "--before",
        type=pd.Timestamp,
        help="Delete only append-dim positions before this timestamp (exclusive).",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dataset = _resolve(args.dataset_id)
    store_factory = dataset.store_factory
    log.info(f"Store: {store_factory.primary_url()}")

    store = store_factory.primary_store()
    root = zarr.open_group(store, mode="r")
    append_dim = dataset.template_config.append_dim
    positions = pd.to_datetime(xr.open_zarr(store, chunks=None)[append_dim].values)

    whole_array = args.lead_index is None and args.before is None
    plan: dict[str, list[str]] = {}
    for path in args.arrays:
        if path not in root:
            sys.exit(f"Not in the store, refusing to run: {path}")
        array = root[path]
        assert isinstance(array, zarr.Array)
        keys = (
            []
            if whole_array
            else list(_chunk_keys(path, array, positions, args.lead_index, args.before))
        )
        plan[path] = keys
        what = "whole array" if whole_array else f"{len(keys)} chunk(s)"
        log.info(f"  {path}: shape {array.shape} chunks {array.chunks} -> {what}")

    total = sum(len(keys) for keys in plan.values())
    if not whole_array and total == 0:
        sys.exit("Selector matched no chunks; nothing to do.")

    if not args.apply:
        log.info(
            f"Dry run. Would delete {'2 array(s)' if whole_array else f'{total} chunk(s)'}"
            f" across {len(plan)} array(s). Pass --apply."
        )
        return

    writable = store_factory.primary_store(writable=True)
    assert isinstance(writable, icechunk.IcechunkStore), type(writable)
    if whole_array:
        write_root = zarr.open_group(writable, mode="a")
        for path in plan:
            del write_root[path]
            log.info(f"Deleted array {path}")
        summary = f"{len(plan)} array(s): {', '.join(plan)}"
    else:
        done = 0
        for keys in plan.values():
            for batch in _batched(keys, _DELETE_BATCH):
                asyncio.run(_delete_all(writable, batch))
                done += len(batch)
                log.info(f"  deleted {done}/{total}")
        summary = f"{total} chunk(s) from {len(plan)} array(s): {', '.join(plan)}"

    snapshot_id = writable.session.commit(
        message=f"Delete {summary}", rebase_with=icechunk.ConflictDetector()
    )
    log.info(f"Committed {snapshot_id}")
    _verify(store_factory, plan, whole_array=whole_array)


def _resolve(dataset_id: str) -> DynamicalDataset[Any, Any]:
    dataset = next((d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id), None)
    if dataset is None:
        known = ", ".join(d.dataset_id for d in DYNAMICAL_DATASETS)
        sys.exit(f"Unknown dataset id {dataset_id!r}. Known: {known}")
    return dataset


def _chunk_keys(
    path: str,
    array: zarr.Array,
    positions: pd.DatetimeIndex,
    lead_index: int | None,
    before: pd.Timestamp | None,
) -> Iterator[str]:
    """Every chunk key of `array` the selectors match.

    The append dim is axis 0 and lead_time, where the array has one, is axis 1.
    """
    grid = [
        math.ceil(size / chunk)
        for size, chunk in zip(array.shape, array.chunks, strict=True)
    ]
    assert grid[0] == len(positions), (path, grid[0], len(positions))
    # A forecast array is (append, lead, y, x); an analysis one (append, y, x). Either
    # way the spatial axes hold one chunk each, so only the leading axes vary.
    has_lead = len(grid) == 4
    assert all(size == 1 for size in grid[2 if has_lead else 1 :]), (path, grid)

    append_indices = range(
        len(positions) if before is None else int(positions.searchsorted(before))
    )
    if lead_index is None:
        lead_indices = list(range(grid[1])) if has_lead else []
    else:
        assert has_lead, f"{path} has no lead_time axis to index"
        assert 0 <= lead_index < grid[1], (path, lead_index, grid[1])
        lead_indices = [lead_index]

    trailing = "/".join("0" * (len(grid) - (2 if has_lead else 1)))
    for i in append_indices:
        for lead in lead_indices or [None]:
            leading = f"{i}" if lead is None else f"{i}/{lead}"
            yield f"{path}/c/{leading}/{trailing}"


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def _delete_all(store: icechunk.IcechunkStore, keys: list[str]) -> None:
    await asyncio.gather(*(store.delete(key) for key in keys))


def _verify(
    store_factory: StoreFactory, plan: dict[str, list[str]], *, whole_array: bool
) -> None:
    root = zarr.open_group(store_factory.primary_store(), mode="r")
    for path, keys in plan.items():
        if whole_array:
            assert path not in root, f"{path} still present"
            continue
        array = root[path]
        assert isinstance(array, zarr.Array)
        assert not asyncio.run(_any_exists(array.store, keys[:50])), (
            f"{path}: sampled keys still present after commit"
        )
    log.info("Verified removed.")


async def _any_exists(store: Store, keys: list[str]) -> bool:
    return any(await asyncio.gather(*(store.exists(key) for key in keys)))


if __name__ == "__main__":
    main()
