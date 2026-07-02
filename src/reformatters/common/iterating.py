import hashlib
import math
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import batched, islice, pairwise, product, starmap
from typing import Any, Literal

import xarray as xr


def node_group_name(node: xr.DataTree) -> str | None:
    """The group name of a DataTree node, or None for the root."""
    return None if node.path == "/" else node.path.removeprefix("/")


def node_path_prefix(node: xr.DataTree) -> str:
    """A node's prefix for building var/coord paths: "" for root, else "group/"."""
    group = node_group_name(node)
    return "" if group is None else f"{group}/"


def walk_data_arrays(tree: xr.DataTree) -> Iterator[tuple[str, xr.DataArray]]:
    """Yield (var_path, DataArray) for every data var across all of a template's groups."""
    for node in tree.subtree:
        prefix = node_path_prefix(node)
        for name, data_array in node.to_dataset().data_vars.items():
            yield f"{prefix}{name}", data_array


def flatten_groups(tree: xr.DataTree) -> xr.Dataset:
    """Flatten a (possibly multi-group, possibly nested) DataTree into one Dataset whose
    data vars are keyed by group path — root vars keep their bare name, a group var
    becomes ``group/name`` (nested: ``a/b/name``), the same path `walk_data_arrays` and
    the zarr store use. Shared coords (duplicated identically into every group) merge;
    each group's own dim coords come along. A single-node tree returns its root dataset.

    Recursing over ``tree.subtree`` means arbitrarily nested groups flatten with no
    special-casing — the node path is the var-name prefix."""
    # Start from the root dataset so its attrs (dataset_id, ...) survive the merge.
    flat = tree.to_dataset()
    for node in tree.subtree:
        prefix = node_path_prefix(node)
        if not prefix:
            continue  # root is already in flat
        node_ds = node.to_dataset()
        flat = flat.merge(
            node_ds.rename({name: f"{prefix}{name}" for name in node_ds.data_vars}),
            compat="no_conflicts",
        )
    return flat


def dimension_slices(
    ds: xr.Dataset, dim: str, kind: Literal["shards", "chunks"] = "shards"
) -> Sequence[slice]:
    """Returns the integer offset slices which correspond to each shard or chunk along a single `dim` of `ds`."""
    chunk_sizes_set = {
        var.encoding[kind][var.dims.index(dim)] for var in ds.data_vars.values()
    }
    assert len(chunk_sizes_set) == 1, (
        f"Inconsistent {kind} sizes among data variables along dimension ({dim}): {chunk_sizes_set}"
    )
    chunk_size = chunk_sizes_set.pop()
    return chunk_slices(len(ds[dim]), chunk_size)


def shard_slice_indexers(da: xr.DataArray) -> Sequence[tuple[slice, ...]]:
    """
    Returns tuples of integer offset slices which correspond to each shard of `da` across all dimensions.
    Each tuple can be used to index into `da` to extract a shard, e.g. `da[shard_indexer]`.
    """
    dim_slices = map(chunk_slices, da.shape, da.encoding["shards"], strict=True)
    return tuple(product(*dim_slices))


def chunk_slices(size: int, chunk_size: int) -> Sequence[slice]:
    """chunk_slices(5, 2) => slice(0, 2), slice(2, 4), slice(4, 6)"""
    indexes = range(0, size + chunk_size, chunk_size)
    slices = tuple(starmap(slice, pairwise(indexes)))
    assert all(s0.stop == s1.start for s0, s1 in pairwise(slices))
    return slices


def get_worker_jobs[T](
    jobs: Iterable[T], worker_index: int, workers_total: int
) -> Sequence[T]:
    """Returns the subset of `jobs` that worker_index should process if there are workers_total workers."""
    assert worker_index >= 0
    assert workers_total >= 1
    assert worker_index < workers_total
    return tuple(islice(jobs, worker_index, None, workers_total))


def get_contiguous_worker_jobs[T](
    jobs: Sequence[T], worker_index: int, workers_total: int
) -> Sequence[T]:
    """Returns the contiguous block of `jobs` that worker_index should process if there are workers_total workers."""
    assert worker_index >= 0
    assert workers_total >= 1
    assert worker_index < workers_total
    block_size = math.ceil(len(jobs) / workers_total)
    worker_jobs = tuple(
        jobs[worker_index * block_size : (worker_index + 1) * block_size]
    )
    assert len(worker_jobs) > 0, (
        f"Worker {worker_index} of {workers_total} has no jobs: "
        f"{len(jobs)} jobs in blocks of {block_size} run out before this worker index"
    )
    return worker_jobs


def spread_evenly[T](items: Sequence[T]) -> list[T]:
    """Reorder so any prefix samples the whole input roughly uniformly.

    Bit-reversal permutation. Concurrently-running workers occupy a contiguous
    worker-index window, so spreading the append-dim regions this way makes them
    process source files scattered across the range instead of a clustered band,
    avoiding hot-spotting a few object-store prefixes. See "Append dim region
    spreading and worker assignment" in docs/parallel_processing.md.
    """
    n = len(items)
    bits = max(1, (n - 1).bit_length())
    order = sorted(range(n), key=lambda i: int(format(i, f"0{bits}b")[::-1], 2))
    return [items[i] for i in order]


def consume[T](iterator: Iterable[T], n: int | None = None) -> None:
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)


def item[T](iterable: Iterable[T]) -> T:
    """Return the single item from the iterable."""
    iterator = iter(iterable)
    try:
        result = next(iterator)
    except StopIteration:
        raise ValueError("Expected exactly one item, got zero") from None

    try:
        next(iterator)
        raise ValueError("Expected exactly one item, got multiple")
    except StopIteration:
        pass

    return result


def digest(data: Iterable[str], length: int = 8) -> str:
    """Consistent, likely collision-free string digest of one or more strings."""
    message = hashlib.sha256()
    for string in data:
        message.update(string.encode())
    return message.hexdigest()[:length]


def split_groups[T](
    groups: Sequence[Sequence[T]], batch_size: int
) -> Sequence[Sequence[T]]:
    """Splits inner groups into smaller groups of at most batch_size."""
    return [
        tuple(split_group)
        for group in groups
        for split_group in batched(group, batch_size, strict=False)
    ]


def group_by[T](
    items: Iterable[T], key_func: Callable[[T], Any]
) -> Sequence[Sequence[T]]:
    """Group items by a key function."""
    groups: dict[Any, list[T]] = {}
    for item in items:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return tuple(groups.values())
