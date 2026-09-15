"""Trim a CHIRPS store to the latest non-NaN day at an Amazon land point."""

import argparse
from dataclasses import dataclass

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
    sample = (
        ds["precipitation_surface"]
        .isel(time=slice(-90, None))
        .sel(latitude=-1.975, longitude=-60.025, method="nearest")
    )
    log.info(
        f"Reading {sample.sizes['time']} days at latitude={float(sample.latitude)}, "
        f"longitude={float(sample.longitude)} from snapshot {session.snapshot_id}"
    )
    present = np.flatnonzero(~np.isnan(sample.values))
    assert present.size, (
        "No non-NaN precipitation at the land point in the last 90 days; refusing to trim"
    )
    new_size = len(times) - sample.sizes["time"] + int(present[-1]) + 1
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
            f"Trim {dataset_id} tail using land-point availability: {times[-1].isoformat()} -> {times[new_size - 1].isoformat()}"
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
        open_repo(args.store, "secret" if args.commit else "anonymous")
        if args.store.startswith("s3://")
        else icechunk.Repository.open(icechunk.local_filesystem_storage(args.store))
    )
    trim_store(repo, args.dataset_id, commit=args.commit)


if __name__ == "__main__":
    main()
