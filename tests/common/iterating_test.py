from collections.abc import Generator

import numpy as np
import pytest
import xarray as xr

from reformatters.common.iterating import (
    chunk_slices,
    consume,
    digest,
    dimension_slices,
    get_worker_jobs,
    group_by,
    item,
    shard_slice_indexers,
)


def test_group_by_parity_preserves_order() -> None:
    items = [3, 1, 4, 2, 5]
    groups = group_by(items, lambda x: x % 2)
    assert groups == ([3, 1, 5], [4, 2])


def test_group_by_empty_returns_empty_tuple() -> None:
    assert group_by([], lambda x: x) == ()


def test_digest() -> None:
    assert digest(["a", "b", "c"]) == "ba7816bf"


def test_item() -> None:
    assert item([1]) == 1


def test_item_multiple() -> None:
    with pytest.raises(ValueError, match="Expected exactly one item, got multiple"):
        item([1, 2, 3])


def test_item_zero() -> None:
    with pytest.raises(ValueError, match="Expected exactly one item, got zero"):
        item([])


def test_consume() -> None:
    yielded_values = []

    def gen() -> Generator[int, None, None]:
        for i in range(10):
            yielded_values.append(i)
            yield i

    consume(gen())
    assert yielded_values == list(range(10))


def test_consume_n() -> None:
    yielded_values = []

    def gen() -> Generator[int, None, None]:
        for i in range(10):
            yielded_values.append(i)
            yield i

    consume(gen(), n=5)
    assert yielded_values == list(range(5))


def test_chunk_slices() -> None:
    assert chunk_slices(5, 2) == (
        slice(0, 2),
        slice(2, 4),
        slice(4, 6),
    )


def test_chunk_slices_negative_size() -> None:
    assert chunk_slices(-5, 2) == ()


def test_chunk_slices_negative_chunk_size() -> None:
    assert chunk_slices(5, -2) == ()


def test_chunk_slices_zero_chunk_size() -> None:
    with pytest.raises(ValueError, match="range\\(\\) arg 3 must not be zero"):
        chunk_slices(5, 0)


def test_chunk_slices_zero_size() -> None:
    assert chunk_slices(0, 2) == ()


def test_get_worker_jobs() -> None:
    assert get_worker_jobs(range(6), 0, 2) == (0, 2, 4)
    assert get_worker_jobs(range(6), 1, 2) == (1, 3, 5)


def test_get_worker_jobs_negative_worker_index() -> None:
    with pytest.raises(AssertionError):
        get_worker_jobs(range(6), -1, 2)


def test_get_worker_jobs_negative_workers_total() -> None:
    with pytest.raises(AssertionError):
        get_worker_jobs(range(6), 0, -1)


def test_get_worker_jobs_zero_workers_total() -> None:
    with pytest.raises(AssertionError):
        get_worker_jobs(range(6), 0, 0)


def test_get_worker_jobs_worker_index_greater_than_workers_total() -> None:
    with pytest.raises(AssertionError):
        get_worker_jobs(range(6), 2, 2)


def test_get_worker_jobs_worker_index_equal_to_workers_total() -> None:
    with pytest.raises(AssertionError):
        get_worker_jobs(range(6), 2, 2)


def test_get_worker_jobs_empty() -> None:
    assert get_worker_jobs([], 0, 1) == ()


def test_shard_slice_indexers() -> None:
    da = xr.DataArray(np.zeros((10, 10)))
    da.encoding["shards"] = (10, 10)
    assert shard_slice_indexers(da) == ((slice(0, 10), slice(0, 10)),)


def test_shard_slice_indexers_multiple_dimensions() -> None:
    da = xr.DataArray(np.zeros((10, 10, 10)))
    da.encoding["shards"] = (10, 10, 10)
    assert shard_slice_indexers(da) == ((slice(0, 10), slice(0, 10), slice(0, 10)),)


def test_dimension_slices() -> None:
    da = xr.DataArray(np.zeros((10, 10)), dims=["time", "temperature"])
    da.encoding["shards"] = (10, 10)
    ds = xr.Dataset({"data": da})
    ds.encoding["shards"] = (10, 10)
    assert dimension_slices(ds, "time") == (slice(0, 10),)
