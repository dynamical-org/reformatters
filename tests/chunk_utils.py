import math
from collections.abc import Iterable
from typing import cast

import xarray as xr


def _shrink_dim(size: int, chunk: int, shard: int) -> tuple[int, int]:
    """
    Smallest (chunk, shard) that tile `size` with the same chunk and shard
    counts as production at this `size`, each capped at two. So a dim that spans
    multiple chunks or multiple shards stays that way, and a single one stays
    single, while the counts (and thus the write cost) stay small.
    """
    assert shard % chunk == 0, f"shard {shard} is not a multiple of chunk {chunk}"

    # n_chunks >= n_shards always, so capping both at 2 keeps n_chunks >= n_shards.
    chunks_target = min(math.ceil(size / chunk), 2)
    shards_target = min(math.ceil(size / shard), 2)

    new_chunk = math.ceil(size / chunks_target)
    new_shard = new_chunk * (chunks_target // shards_target)
    return new_chunk, new_shard


def shrink_chunks_and_shards[T: (xr.Dataset, xr.DataTree)](
    ds: T, dims: Iterable[str] | None = None
) -> T:
    """
    Shrink every data var's chunk and shard encoding to the smallest layout that
    still exercises the same structure as production at this dataset's sizes.

    Accepts a template as either a Dataset or a DataTree (per-node shrink).

    Integration tests trim the append/lead/ensemble dims then write a small
    region. With production chunk geometry that means allocating, filling, and
    compressing a full production-sized chunk buffer for every chunk touched
    (often thousands of near-empty chunks), which dominates test runtime without
    adding coverage.

    For each dim, this keeps the production chunk and shard counts at the current
    size, capped at two: a dim that spans multiple chunks or multiple shards in
    production stays multi, and a single one stays single, so the shrink cannot
    accidentally collapse a chunk or shard boundary a test depends on. (It does
    not preserve how many chunks pack into a shard, which is zarr-internal.)

    Call this *after* trimming the template (e.g. after `.sel(lead_time=...)`) so
    the sizes reflect what the test actually writes. Pass `dims` to shrink only
    those dimensions, e.g. shrink the spatial dims while leaving the append dim
    at production size for a shard-boundary test.
    """
    if isinstance(ds, xr.DataTree):
        return cast(
            "T",
            xr.DataTree.from_dict(
                {
                    node.path: shrink_chunks_and_shards(node.to_dataset(), dims)
                    for node in ds.subtree
                }
            ),
        )

    selected = set(dims) if dims is not None else None
    for var in ds.data_vars.values():
        chunks = list(var.encoding["chunks"])
        has_shards = var.encoding.get("shards") is not None
        shards = list(var.encoding["shards"]) if has_shards else list(chunks)
        for axis, dim in enumerate(var.dims):
            if selected is not None and dim not in selected:
                continue
            chunks[axis], shards[axis] = _shrink_dim(
                ds.sizes[dim], chunks[axis], shards[axis]
            )
        var.encoding["chunks"] = tuple(chunks)
        if has_shards:
            var.encoding["shards"] = tuple(shards)
    return ds
