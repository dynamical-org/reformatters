"""Remove an entirely NaN tail from a CHIRPS Icechunk store."""

import argparse
from dataclasses import dataclass
from itertools import product

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common.logging import get_logger
from scripts.icechunk_utils import open_repo

log = get_logger(__name__)
DATASET_IDS = (
    "ucsb-chc-chirps-analysis-final",
    "ucsb-chc-chirps-analysis-preliminary",
)


@dataclass(frozen=True)
class TrimResult:
    before_snapshot: str
    after_snapshot: str | None
    before_size: int
    after_size: int
    start: pd.Timestamp
    before_end: pd.Timestamp
    after_end: pd.Timestamp


def last_non_nan_index(array: zarr.Array, time_axis: int) -> int | None:
    """Find the last time with any non-NaN cell, reading at most one logical chunk at a time."""
    assert np.issubdtype(array.dtype, np.floating), array.path
    assert np.isnan(array.fill_value), f"{array.path}: expected NaN fill value"
    size = array.shape[time_axis]
    time_chunk = array.chunks[time_axis]
    spatial_axes = [axis for axis in range(array.ndim) if axis != time_axis]
    spatial_starts = [
        range(0, array.shape[axis], array.chunks[axis]) for axis in spatial_axes
    ]
    for start in reversed(range(0, size, time_chunk)):
        stop = min(start + time_chunk, size)
        last = None
        for starts in product(*spatial_starts):
            selection = [
                slice(start, stop) if axis == time_axis else slice(None)
                for axis in range(array.ndim)
            ]
            for axis, offset in zip(spatial_axes, starts, strict=True):
                selection[axis] = slice(
                    offset, min(offset + array.chunks[axis], array.shape[axis])
                )
            values = np.asarray(array[tuple(selection)])
            present = np.flatnonzero(
                np.any(~np.isnan(values), axis=tuple(spatial_axes))
            )
            if present.size:
                last = max(
                    last if last is not None else start, start + int(present[-1])
                )
                if last == stop - 1:
                    return last
        if last is not None:
            return last
    return None


def trim_store(
    repo: icechunk.Repository, dataset_id: str, *, commit: bool = False
) -> TrimResult:
    session = repo.readonly_session("main")
    root = zarr.open_group(session.store, mode="r", use_consolidated=False)
    assert dataset_id in DATASET_IDS
    assert root.attrs["dataset_id"] == dataset_id
    assert not list(root.groups()), "Expected a single-level CHIRPS store"
    ds = xr.open_zarr(session.store, chunks=None, consolidated=False)
    times = pd.DatetimeIndex(ds.time.values)
    assert len(times)
    assert times.is_monotonic_increasing
    assert times.is_unique
    assert not times.hasnans
    arrays = dict(root.arrays())
    time_axes = {}
    for name, array in arrays.items():
        assert isinstance(array.metadata, ArrayV3Metadata)
        dims = array.metadata.dimension_names
        if dims and "time" in dims:
            time_axes[name] = dims.index("time")
    assert "time" in time_axes
    assert all(
        arrays[name].shape[axis] == len(times) for name, axis in time_axes.items()
    )
    last_indices = []
    for name, var in ds.data_vars.items():
        if "time" not in var.dims:
            continue
        log.info(
            f"Scanning {dataset_id}/{name} from the end of snapshot {session.snapshot_id}"
        )
        last = last_non_nan_index(arrays[str(name)], time_axes[str(name)])
        if last is not None:
            last_indices.append(last)
    assert last_indices, "No non-NaN data found; refusing to empty the store"
    new_size = max(last_indices) + 1
    after_snapshot = None
    if commit and new_size < len(times):
        writable = repo.writable_session("main")
        assert writable.snapshot_id == session.snapshot_id, (
            "main changed during the scan; rerun"
        )
        writable_root = zarr.open_group(
            writable.store, mode="r+", use_consolidated=False
        )
        for name, axis in time_axes.items():
            array = writable_root[name]
            assert isinstance(array, zarr.Array)
            shape = list(array.shape)
            shape[axis] = new_size
            array.resize(tuple(shape))
        # No rebase: a concurrent update invalidates the scanned extent.
        after_snapshot = writable.commit(
            f"Trim {dataset_id} all-NaN tail: {times[-1].isoformat()} -> {times[new_size - 1].isoformat()}"
        )
    result = TrimResult(
        session.snapshot_id,
        after_snapshot,
        len(times),
        new_size,
        times[0],
        times[-1],
        times[new_size - 1],
    )
    action = (
        "Committed"
        if after_snapshot
        else "No change"
        if new_size == len(times)
        else "Dry run"
    )
    log.info(
        f"{action}: {dataset_id}: {result.start.isoformat()} through {result.before_end.isoformat()} "
        f"({result.before_size} days) -> {result.start.isoformat()} through {result.after_end.isoformat()} "
        f"({result.after_size} days); snapshot {result.before_snapshot} -> {result.after_snapshot or result.before_snapshot}"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id", choices=DATASET_IDS)
    parser.add_argument("store", help="Explicit S3 URI or local Icechunk directory")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit the trim to main (default: dry run)",
    )
    args = parser.parse_args(argv)
    repo = (
        open_repo(args.store, "env" if args.commit else "anonymous")
        if args.store.startswith("s3://")
        else icechunk.Repository.open(icechunk.local_filesystem_storage(args.store))
    )
    trim_store(repo, args.dataset_id, commit=args.commit)


if __name__ == "__main__":
    main()
