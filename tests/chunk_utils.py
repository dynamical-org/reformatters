from collections.abc import Mapping

import xarray as xr


def shrink_chunks_and_shards(
    ds: xr.Dataset, sizes: Mapping[str, tuple[int, int]]
) -> xr.Dataset:
    """
    Override the (chunk, shard) sizes of all data vars along the dims in `sizes`.

    Integration tests write small, dim-trimmed regions. At that scale the
    production chunk geometry means writing thousands of nearly-empty chunks
    (zarr materializes and compresses a full chunk buffer for each partial
    chunk write), which dominates test runtime without adding coverage.
    Tests use this to keep a multi-chunk, multi-shard layout while writing
    only a handful of chunks per variable.
    """
    for var in ds.data_vars.values():
        chunks = list(var.encoding["chunks"])
        shards = list(var.encoding["shards"])
        for dim, (chunk, shard) in sizes.items():
            assert shard % chunk == 0, f"shard {shard} not a multiple of chunk {chunk}"
            axis = var.dims.index(dim)
            chunks[axis] = chunk
            shards[axis] = shard
        var.encoding["chunks"] = tuple(chunks)
        var.encoding["shards"] = tuple(shards)
    return ds
