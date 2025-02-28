from collections import deque
from collections.abc import Iterable
from itertools import islice, pairwise, product, starmap
from typing import Literal

import xarray as xr


def dimension_slices(
    ds: xr.Dataset, dim: str, kind: Literal["shards", "chunks"] = "shards"
) -> Iterable[slice]:
    """Returns the integer offset slices which correspond to each shard or chunk along a single `dim` of `ds`."""
    chunk_sizes_set = {
        var.encoding[kind][var.dims.index(dim)] for var in ds.data_vars.values()
    }
    assert len(chunk_sizes_set) == 1, (
        f"Inconsistent {kind} sizes among data variables along dimension ({dim}): {chunk_sizes_set}"
    )
    chunk_size = chunk_sizes_set.pop()
    return chunk_slices(len(ds[dim]), chunk_size)


def shard_slice_indexers(da: xr.DataArray) -> Iterable[tuple[slice, ...]]:
    """
    Returns tuples of integer offset slices which correspond to each shard of `da` across all dimensions.
    Each tuple can be used to index into `da` to extract a shard, e.g. `da[shard_indexer]`.
    """
    dim_slices = map(chunk_slices, da.shape, da.encoding["shards"])
    return product(*dim_slices)


def chunk_slices(size: int, chunk_size: int) -> Iterable[slice]:
    """chunk_slices(5, 2) => slice(0, 2), slice(2, 4), slice(4, 6)"""
    indexes = range(0, size + chunk_size, chunk_size)
    return starmap(slice, pairwise(indexes))


def get_worker_jobs[T](
    jobs: Iterable[T], worker_index: int, workers_total: int
) -> Iterable[T]:
    """Returns the subset of `jobs` that worker_index should process if there are workers_total workers."""
    assert worker_index >= 0 and workers_total >= 1
    assert worker_index < workers_total
    return islice(jobs, worker_index, None, workers_total)


def consume[T](iterator: Iterable[T], n: int | None = None) -> None:
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)
