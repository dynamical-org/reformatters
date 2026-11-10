import math
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from reformatters.common import materialized_region_job, shared_memory_utils
from tests.chunk_utils import (
    _shrink_dim,
    patch_write_shards_containing,
    shrink_chunks_and_shards,
)


def _structure(size: int, chunk: int, shard: int) -> tuple[bool, bool]:
    """(spans_multiple_chunks, spans_multiple_shards) at `size`."""
    return math.ceil(size / chunk) >= 2, math.ceil(size / shard) >= 2


@pytest.mark.parametrize(
    ("size", "chunk", "shard"),
    [
        (2, 64, 192),  # bloated trimmed dim -> single chunk/shard
        (3, 1440, 2880),  # bloated trimmed time dim -> single
        (721, 17, 374),  # spatial: multi chunk, multi shard, multi chunk/shard
        (1440, 16, 368),  # spatial: all multi, exact tiling
        (721, 361, 361),  # multi shard, one chunk per shard
        (80, 30, 90),  # single shard, multiple chunks per shard
        (50, 50, 50),  # exactly one full chunk/shard
        (51, 50, 50),  # tiny partial second chunk -> multi chunk & shard
    ],
)
def test_shrink_dim_preserves_structure_and_shrinks(
    size: int, chunk: int, shard: int
) -> None:
    new_chunk, new_shard = _shrink_dim(size, chunk, shard)

    assert new_shard % new_chunk == 0  # zarr requires whole chunks per shard
    assert new_chunk <= size  # never allocate a chunk larger than the data

    assert _structure(size, new_chunk, new_shard) == _structure(size, chunk, shard)

    # counts capped at two so cost stays small
    assert math.ceil(size / new_shard) <= 2
    assert new_shard // new_chunk <= 2


def _make_dataset(
    sizes: dict[str, int],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> xr.Dataset:
    dims = tuple(sizes)
    da = xr.DataArray(np.zeros(tuple(sizes.values()), dtype=np.float32), dims=dims)
    da.encoding = {"chunks": chunks}
    if shards is not None:
        da.encoding["shards"] = shards
    return xr.Dataset({"v": da})


def test_shrink_dataset_all_dims() -> None:
    sizes = {"time": 3, "latitude": 721, "longitude": 1440}
    ds = _make_dataset(sizes, (1440, 17, 16), (2880, 374, 368))

    shrink_chunks_and_shards(ds)
    enc = ds["v"].encoding

    # time collapses to a single small chunk
    assert enc["chunks"][0] <= 3
    assert enc["shards"][0] <= 3
    # spatial dims keep their multi-shard structure
    assert math.ceil(721 / enc["shards"][1]) == 2
    assert math.ceil(1440 / enc["shards"][2]) == 2
    assert all(s % c == 0 for c, s in zip(enc["chunks"], enc["shards"], strict=True))


def test_shrink_dataset_dims_subset_leaves_others_untouched() -> None:
    sizes = {"time": 3, "latitude": 721, "longitude": 1440}
    ds = _make_dataset(sizes, (1440, 17, 16), (2880, 374, 368))

    shrink_chunks_and_shards(ds, dims=["latitude", "longitude"])
    enc = ds["v"].encoding

    # time left at production geometry
    assert enc["chunks"][0] == 1440
    assert enc["shards"][0] == 2880
    # spatial dims shrunk to two chunks each, structure preserved
    assert math.ceil(721 / enc["chunks"][1]) == 2
    assert math.ceil(1440 / enc["chunks"][2]) == 2


def test_shrink_dataset_without_shards() -> None:
    sizes = {"time": 3, "latitude": 721, "longitude": 1440}
    ds = _make_dataset(sizes, (1440, 17, 16), None)

    shrink_chunks_and_shards(ds)

    assert "shards" not in ds["v"].encoding
    assert ds["v"].encoding["chunks"][0] <= 3


@pytest.mark.parametrize("second_cell", [False, True])
def test_patch_write_shards_containing_selects_distinct_shards(
    monkeypatch: pytest.MonkeyPatch, second_cell: bool
) -> None:
    ds = _make_dataset(
        {"time": 2, "latitude": 3, "longitude": 5}, (1, 1, 1), (2, 2, 2)
    ).assign_coords(latitude=[30.0, 20.0, 10.0], longitude=[-2, -1, 0, 1, 2])
    ds = ds.rename({"v": "pressure_level/temperature"})
    write = Mock()
    monkeypatch.setattr(shared_memory_utils, "write_shard_to_zarr", write)
    cells = [{"latitude": 10.1, "longitude": 1.9}] * 2
    expected = [(slice(0, 2), slice(2, 4), slice(4, 6))]
    if second_cell:
        cells.append({"latitude": 29.9, "longitude": -1.9})
        expected.insert(0, (slice(0, 2), slice(0, 2), slice(0, 2)))
    patch_write_shards_containing(monkeypatch, cells)
    shared_buffer = Mock()
    shared_buffer.name = "test-buffer"

    materialized_region_job.write_shards(
        ds["pressure_level/temperature"],
        shared_buffer,
        "time",
        ds,
        Mock(),
        Mock(),
    )

    assert [call.args[-1] for call in write.call_args_list] == expected
    assert all(call.args[1] == "test-buffer" for call in write.call_args_list)
