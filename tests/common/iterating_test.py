import math
from collections.abc import Generator
from itertools import pairwise

import numpy as np
import pytest
import xarray as xr

from reformatters.common.iterating import (
    chunk_slices,
    consume,
    digest,
    dimension_slices,
    flatten_groups,
    get_contiguous_worker_jobs,
    get_worker_jobs,
    group_by,
    item,
    node_group_name,
    node_path_prefix,
    shard_slice_indexers,
    spread_evenly,
    walk_data_arrays,
)


def _two_node_tree() -> xr.DataTree:
    root = xr.Dataset(
        {"temperature_2m": xr.Variable(("time",), np.zeros(2, dtype=np.float32))},
        coords={"time": [0, 1]},
        attrs={"dataset_id": "test-dataset"},
    )
    pressure = xr.Dataset(
        {
            "temperature": xr.Variable(
                ("time", "pressure_level"), np.zeros((2, 3), dtype=np.float32)
            )
        },
        coords={"time": [0, 1], "pressure_level": [1000.0, 850.0, 500.0]},
    )
    return xr.DataTree.from_dict({"/": root, "/pressure_level": pressure})


def test_node_group_name_root_and_child() -> None:
    root, pressure = _two_node_tree().subtree
    assert node_group_name(root) is None
    assert node_group_name(pressure) == "pressure_level"


def test_node_path_prefix_root_and_child() -> None:
    root, pressure = _two_node_tree().subtree
    assert node_path_prefix(root) == ""
    assert node_path_prefix(pressure) == "pressure_level/"


def test_walk_data_arrays_yields_group_qualified_paths() -> None:
    tree = _two_node_tree()
    paths = [path for path, _ in walk_data_arrays(tree)]
    assert paths == ["temperature_2m", "pressure_level/temperature"]
    by_path = dict(walk_data_arrays(tree))
    assert by_path["pressure_level/temperature"].dims == ("time", "pressure_level")


def test_flatten_groups_single_node_returns_root() -> None:
    root = xr.Dataset(
        {"temperature_2m": xr.Variable(("time",), np.zeros(2, dtype=np.float32))},
        coords={"time": [0, 1]},
        attrs={"dataset_id": "x"},
    )
    flat = flatten_groups(xr.DataTree.from_dict({"/": root}))
    assert list(flat.data_vars) == ["temperature_2m"]
    assert flat.identical(root)  # identical (not equals) also checks attrs


def test_flatten_groups_keys_group_vars_by_path() -> None:
    flat = flatten_groups(_two_node_tree())
    assert set(flat.data_vars) == {"temperature_2m", "pressure_level/temperature"}
    # The group var keeps its vertical dim and that group's own coord is merged in.
    assert flat["pressure_level/temperature"].dims == ("time", "pressure_level")
    assert "pressure_level" in flat.coords
    # Root attrs (read by check_for_expected_shards) survive the flatten/merge.
    assert flat.attrs["dataset_id"] == "test-dataset"


def test_flatten_groups_handles_arbitrary_nesting() -> None:
    # Nested groups flatten by node path with no special-casing.
    coords = {"x": [0, 1]}
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset({"a": ("x", [1.0, 2.0])}, coords=coords),
            "/g": xr.Dataset({"b": ("x", [3.0, 4.0])}, coords=coords),
            "/g/h": xr.Dataset({"c": ("x", [5.0, 6.0])}, coords=coords),
        }
    )
    flat = flatten_groups(tree)
    assert set(flat.data_vars) == {"a", "g/b", "g/h/c"}


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

    def gen() -> Generator[int]:
        for i in range(10):
            yielded_values.append(i)
            yield i

    consume(gen())
    assert yielded_values == list(range(10))


def test_consume_n() -> None:
    yielded_values = []

    def gen() -> Generator[int]:
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


def test_get_contiguous_worker_jobs() -> None:
    assert get_contiguous_worker_jobs(list(range(6)), 0, 2) == (0, 1, 2)
    assert get_contiguous_worker_jobs(list(range(6)), 1, 2) == (3, 4, 5)
    assert get_contiguous_worker_jobs(list(range(7)), 2, 3) == (6,)


def test_get_contiguous_worker_jobs_partitions_all_jobs_in_order() -> None:
    # Callers derive workers_total = ceil(n_jobs / jobs_per_pod), which guarantees
    # every worker's block is non-empty.
    for n, jobs_per_pod in ((11641, 30), (7, 3), (100, 7), (5, 5), (1, 1), (30, 30)):
        jobs = list(range(n))
        workers_total = math.ceil(n / jobs_per_pod)
        blocks = [
            get_contiguous_worker_jobs(jobs, worker_index, workers_total)
            for worker_index in range(workers_total)
        ]
        assert all(len(block) > 0 for block in blocks)
        # Blocks are disjoint, cover all jobs exactly once, and preserve order.
        assert [job for block in blocks for job in block] == jobs


def test_get_contiguous_worker_jobs_invalid_worker_index() -> None:
    with pytest.raises(AssertionError):
        get_contiguous_worker_jobs(list(range(6)), -1, 2)
    with pytest.raises(AssertionError):
        get_contiguous_worker_jobs(list(range(6)), 2, 2)


def test_get_contiguous_worker_jobs_empty_block_raises() -> None:
    # 5 jobs / 4 workers -> blocks of 2; worker 3's block would be empty.
    with pytest.raises(AssertionError, match="has no jobs"):
        get_contiguous_worker_jobs(list(range(5)), 3, 4)


def test_spread_evenly_is_a_permutation() -> None:
    for n in (0, 1, 2, 7, 8, 100, 8326):
        result = spread_evenly(list(range(n)))
        assert sorted(result) == list(range(n))
        assert result == spread_evenly(list(range(n)))  # deterministic


def test_spread_evenly_preserves_elements_not_just_indices() -> None:
    items = ["a", "b", "c", "d"]
    assert sorted(spread_evenly(items)) == sorted(items)


def test_spread_evenly_any_prefix_covers_the_full_range() -> None:
    # Any contiguous worker-index window must span the input, not cluster in a
    # band. Check the first `window` entries are spread within ~3x of uniform.
    n = 8326
    result = spread_evenly(list(range(n)))
    for window in (64, 128, 256):
        head = sorted(result[:window])
        max_gap = max(b - a for a, b in pairwise(head))
        assert max_gap < 3 * (n // window)
        assert head[0] < n // window
        assert head[-1] > n - n // window


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
