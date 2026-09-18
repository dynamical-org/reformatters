from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from typer.testing import CliRunner
from zarr.abc.store import Store
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from reformatters.common import template_utils
from reformatters.common.chunk_removal import chunk_keys, remove_from_stores
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
    StoreFactory,
    commit_if_icechunk,
)
from tests.common.virtual_region_job_test import (
    N_LEADS,
    VirtualTestDataset,
    _backfilled_store,
    _block_values,
    _create_template_ds,
    _make_dataset,
    _primary_repo,
    _snapshot_count,
)

INITS = pd.date_range("2018-07-13T12:00", periods=3, freq="6h")
TIMES = pd.date_range("2014-10-01", periods=4, freq="1h")


def _array(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None = None,
) -> zarr.Array:
    return zarr.create_array(
        MemoryStore(),
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype="float32",
        dimension_names=dims,
    )


FORECAST = _array(
    ("init_time", "lead_time", "latitude", "longitude"), (3, 49, 8, 9), (1, 1, 8, 9)
)
ANALYSIS = _array(("time", "latitude", "longitude"), (4, 8, 9), (1, 8, 9))


def test_lead_index_selects_that_lead_at_every_init() -> None:
    assert chunk_keys("p", FORECAST, "init_time", INITS, lead_index=0) == [
        "p/c/0/0/0/0",
        "p/c/1/0/0/0",
        "p/c/2/0/0/0",
    ]


def test_before_selects_every_lead_of_the_earlier_positions() -> None:
    array = _array(
        ("init_time", "lead_time", "latitude", "longitude"), (3, 2, 8, 9), (1, 1, 8, 9)
    )
    assert chunk_keys(
        "v", array, "init_time", INITS, before=pd.Timestamp("2018-07-13T18:00")
    ) == ["v/c/0/0/0/0", "v/c/0/1/0/0"]


def test_an_array_without_a_lead_axis_keys_on_the_append_dim_alone() -> None:
    assert chunk_keys(
        "model_level/tke",
        ANALYSIS,
        "time",
        TIMES,
        before=pd.Timestamp("2014-10-01T02:00"),
    ) == ["model_level/tke/c/0/0/0", "model_level/tke/c/1/0/0"]


def test_at_selects_exactly_those_positions() -> None:
    assert chunk_keys("v", ANALYSIS, "time", TIMES, at=[TIMES[3], TIMES[1]]) == [
        "v/c/1/0/0",
        "v/c/3/0/0",
    ]


def test_at_and_before_combine() -> None:
    assert chunk_keys(
        "v", ANALYSIS, "time", TIMES, at=[TIMES[3], TIMES[1]], before=TIMES[2]
    ) == ["v/c/1/0/0"]


def test_lead_index_on_an_ensemble_array_spans_every_member() -> None:
    array = _array(
        ("init_time", "ensemble_member", "lead_time", "latitude", "longitude"),
        (3, 2, 4, 8, 9),
        (1, 1, 1, 8, 9),
    )
    assert chunk_keys("v", array, "init_time", INITS, lead_index=3, at=[INITS[1]]) == [
        "v/c/1/0/3/0/0",
        "v/c/1/1/3/0/0",
    ]


def test_sharded_array_keys_every_spatial_shard_of_a_fully_selected_shard() -> None:
    array = _array(
        ("time", "latitude", "longitude"), (4, 8, 9), (1, 4, 3), shards=(2, 8, 3)
    )
    assert chunk_keys("v", array, "time", TIMES, at=[TIMES[2], TIMES[3]]) == [
        "v/c/1/0/0",
        "v/c/1/0/1",
        "v/c/1/0/2",
    ]


def test_sharded_array_refuses_a_partially_selected_shard() -> None:
    array = _array(
        ("time", "latitude", "longitude"), (4, 8, 9), (1, 4, 3), shards=(2, 8, 3)
    )
    with pytest.raises(AssertionError, match="covers only some"):
        chunk_keys("v", array, "time", TIMES, at=[TIMES[2]])


def test_lead_index_refuses_a_shard_spanning_leads() -> None:
    array = _array(
        ("init_time", "lead_time", "latitude", "longitude"),
        (3, 4, 8, 9),
        (1, 1, 8, 9),
        shards=(1, 2, 8, 9),
    )
    with pytest.raises(AssertionError, match="cannot delete one alone"):
        chunk_keys("v", array, "init_time", INITS, lead_index=1)


def test_at_must_be_in_the_record() -> None:
    with pytest.raises(AssertionError, match="not in the record"):
        chunk_keys("v", ANALYSIS, "time", TIMES, at=[pd.Timestamp("2000-01-01")])


def test_before_the_whole_record_selects_nothing() -> None:
    assert chunk_keys("v", ANALYSIS, "time", TIMES, before=TIMES[0]) == []


def test_lead_index_requires_a_lead_axis() -> None:
    with pytest.raises(AssertionError, match="no lead_time axis"):
        chunk_keys("v", ANALYSIS, "time", TIMES, lead_index=0)


def test_lead_index_must_be_in_range() -> None:
    with pytest.raises(AssertionError):
        chunk_keys("v", FORECAST, "init_time", INITS, lead_index=49)


def test_positions_must_match_the_array() -> None:
    with pytest.raises(AssertionError):
        chunk_keys("v", FORECAST, "init_time", TIMES, lead_index=0)


def test_append_dim_must_be_the_first_axis() -> None:
    with pytest.raises(AssertionError):
        chunk_keys("v", ANALYSIS, "init_time", TIMES, lead_index=0)


# ---- Against a local icechunk repo ----


def _exists(store: Store, key: str) -> bool:
    return sync(store.exists(key))


def _values(dataset: VirtualTestDataset) -> xr.DataArray:
    return xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )["temperature_2m"]


@pytest.fixture
def backfilled(tmp_path: Path) -> VirtualTestDataset:
    dataset = _make_dataset(tmp_path)
    _backfilled_store(dataset, _create_template_ds(4), emit=slice(0, 4))
    return dataset


def test_dry_run_changes_nothing(backfilled: VirtualTestDataset) -> None:
    repo = _primary_repo(backfilled.store_factory)
    snapshots = _snapshot_count(repo)
    remove_from_stores(
        backfilled.store_factory, "init_time", ["temperature_2m"], lead_index=0
    )
    assert _snapshot_count(repo) == snapshots
    assert not _values(backfilled).isnull().any()


def test_lead_index_removal_leaves_other_leads_and_commits_once(
    backfilled: VirtualTestDataset,
) -> None:
    repo = _primary_repo(backfilled.store_factory)
    snapshots = _snapshot_count(repo)
    remove_from_stores(
        backfilled.store_factory,
        "init_time",
        ["temperature_2m"],
        lead_index=0,
        apply=True,
    )
    assert _snapshot_count(repo) == snapshots + 1
    assert next(iter(repo.ancestry(branch="main"))).message == (
        "Delete 4 chunk(s) from 1 array(s): temperature_2m"
    )
    values = _values(backfilled)
    assert values.isel(lead_time=0).isnull().all()
    for init_idx in range(4):
        for lead_idx in range(1, N_LEADS):
            np.testing.assert_array_equal(
                values.isel(init_time=init_idx, lead_time=lead_idx).values,
                _block_values(init_idx, lead_idx),
            )


def test_at_removal_takes_every_lead_of_those_positions(
    backfilled: VirtualTestDataset,
) -> None:
    inits = pd.DatetimeIndex(_values(backfilled).init_time.values)
    remove_from_stores(
        backfilled.store_factory,
        "init_time",
        ["temperature_2m"],
        at=[inits[1], inits[3]],
        apply=True,
    )
    values = _values(backfilled)
    assert values.isel(init_time=[1, 3]).isnull().all()
    assert not values.isel(init_time=[0, 2]).isnull().any()


def test_whole_array_removal(backfilled: VirtualTestDataset) -> None:
    remove_from_stores(
        backfilled.store_factory,
        "init_time",
        ["temperature_2m"],
        whole_array=True,
        apply=True,
    )
    root = zarr.open_group(backfilled.store_factory.primary_store(), mode="r")
    assert "temperature_2m" not in root
    assert "init_time" in root


def test_refuses_an_array_not_in_the_store(backfilled: VirtualTestDataset) -> None:
    with pytest.raises(AssertionError, match="Not in the store"):
        remove_from_stores(
            backfilled.store_factory, "init_time", ["nope"], whole_array=True
        )


def test_requires_exactly_one_kind_of_selector(backfilled: VirtualTestDataset) -> None:
    with pytest.raises(AssertionError, match="not both"):
        remove_from_stores(backfilled.store_factory, "init_time", ["temperature_2m"])
    with pytest.raises(AssertionError, match="not both"):
        remove_from_stores(
            backfilled.store_factory,
            "init_time",
            ["temperature_2m"],
            lead_index=0,
            whole_array=True,
        )


@pytest.fixture
def with_replica(tmp_path: Path) -> StoreFactory:
    """A primary icechunk store plus a zarr v3 replica, both carrying the array's
    metadata and one written chunk at the first position."""
    factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        replica_storage_configs=[
            StorageConfig(base_path=str(tmp_path), format=DatasetFormat.ZARR3)
        ],
        dataset_id="test-replica-removal",
        template_config_version="v1.0",
    )
    template_utils.write_metadata(_create_template_ds(4), factory)
    for store in [
        factory.primary_store(writable=True),
        *factory.replica_stores(writable=True),
    ]:
        array = zarr.open_group(store, mode="a")["temperature_2m"]
        assert isinstance(array, zarr.Array)
        array[0, 0] = 1.0
        commit_if_icechunk("chunk", store, [])
    return factory


def test_chunk_removal_reaches_every_replica(with_replica: StoreFactory) -> None:
    factory = with_replica
    stores = [factory.primary_store(), *factory.replica_stores()]
    assert len(stores) == 2
    for store in stores:
        assert _exists(store, "temperature_2m/c/0/0/0/0")

    remove_from_stores(
        factory, "init_time", ["temperature_2m"], lead_index=0, apply=True
    )
    for store in [factory.primary_store(), *factory.replica_stores()]:
        assert not _exists(store, "temperature_2m/c/0/0/0/0")


def test_whole_array_removal_refreshes_replica_consolidated_metadata(
    with_replica: StoreFactory,
) -> None:
    factory = with_replica
    remove_from_stores(
        factory, "init_time", ["temperature_2m"], whole_array=True, apply=True
    )
    for store in [factory.primary_store(), *factory.replica_stores()]:
        root = zarr.open_group(store, mode="r")
        assert "temperature_2m" not in root
        assert "init_time" in root
    (replica,) = factory.replica_stores()
    consolidated = zarr.open_group(replica, mode="r").metadata.consolidated_metadata
    assert consolidated is not None
    assert "temperature_2m" not in consolidated.metadata


def test_cli_parses_selectors_and_is_a_dry_run_by_default(
    backfilled: VirtualTestDataset,
) -> None:
    runner = CliRunner()
    app = backfilled.get_cli()
    args = ["remove-chunks", "temperature_2m", "--at", "2024-01-01T06:00:00"]

    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert not _values(backfilled).isnull().any()

    result = runner.invoke(app, [*args, "--at", "2024-01-01T18:00:00", "--apply"])
    assert result.exit_code == 0, result.output
    values = _values(backfilled)
    assert values.isel(init_time=[1, 3]).isnull().all()
    assert not values.isel(init_time=[0, 2]).isnull().any()

    result = runner.invoke(app, ["remove-chunks", "temperature_2m"])
    assert result.exit_code != 0
    assert isinstance(result.exception, AssertionError)
