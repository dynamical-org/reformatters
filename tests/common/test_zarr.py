from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
import pytest
import xarray as xr
import zarr.storage
from icechunk.store import IcechunkStore
from zarr.abc.store import Store

from reformatters.common import zarr as zarr_module
from reformatters.common.zarr import (
    assert_fill_values_set,
    copy_data_var,
    copy_zarr_metadata,
    sync_to_store,
)


@pytest.fixture
def template_ds() -> xr.Dataset:
    return xr.Dataset(
        coords={
            "time": ["2000-01-01", "2000-01-02"],
            "lat": [10, 20, 30],
            "lon": [100, 110, 120],
        }
    )


@pytest.fixture
def tmp_store_and_metadata_files(tmp_path: Path) -> tuple[Path, list[Path]]:
    store_path = tmp_path / "tmp_store"
    store_path.mkdir()

    # Create coordinate directories first
    for coord in ["time", "lat", "lon"]:
        (store_path / coord).mkdir()

    # Create mock metadata files structure
    (store_path / "zarr.json").touch()
    (store_path / "time" / "zarr.json").touch()
    (store_path / "lat" / "zarr.json").touch()
    (store_path / "lon" / "zarr.json").touch()

    # Create coordinate chunk files
    for coord in ["time", "lat", "lon"]:
        coord_dir = store_path / coord / "c"
        coord_dir.mkdir(parents=True)
        (coord_dir / "0").touch()

    # Coordinate label arrays should be copied before metadata
    metadata_files: list[Path] = []
    for coord in ["time", "lat", "lon"]:
        metadata_files.extend(
            f for f in store_path.glob(f"{coord}/c/**/*") if f.is_file()
        )
    metadata_files.append(store_path / "zarr.json")
    metadata_files.extend(store_path.glob("*/zarr.json"))

    return store_path, metadata_files


def test_copy_zarr_metadata_calls_copy_metadata_files_for_all_stores(
    monkeypatch: pytest.MonkeyPatch,
    template_ds: xr.Dataset,
    tmp_store_and_metadata_files: tuple[Path, list[Path]],
) -> None:
    tmp_store, expected_metadata_files = tmp_store_and_metadata_files

    mock_copy_metadata_files = Mock()
    monkeypatch.setattr(zarr_module, "_copy_metadata_files", mock_copy_metadata_files)

    mock_primary_store = Mock(spec=Store)
    mock_replica_store_zarr = Mock(spec=Store)
    mock_replica_store_icechunk = Mock(spec=IcechunkStore)
    replica_stores = [mock_replica_store_zarr, mock_replica_store_icechunk]

    copy_zarr_metadata(template_ds, tmp_store, mock_primary_store, replica_stores)

    # Should be called once for each replica store, then once for primary store
    assert mock_copy_metadata_files.call_count == 3
    calls = mock_copy_metadata_files.call_args_list

    # Instead of exact call matching, check the calls contain the right elements
    assert len(calls) == 3
    for call_args in calls:
        files, _store_path, _store = call_args[0]
        # Verify the files list contains the expected files (order-independent)
        assert set(files) == set(expected_metadata_files)


def test_copy_zarr_metadata_skips_non_icechunk_stores_when_icechunk_only(
    monkeypatch: pytest.MonkeyPatch,
    template_ds: xr.Dataset,
    tmp_store_and_metadata_files: tuple[Path, list[Path]],
) -> None:
    tmp_store, expected_metadata_files = tmp_store_and_metadata_files

    mock_copy_metadata_files = Mock()
    monkeypatch.setattr(zarr_module, "_copy_metadata_files", mock_copy_metadata_files)

    mock_primary_icechunk = Mock(spec=IcechunkStore)
    mock_replica_store_icechunk = Mock(spec=IcechunkStore)
    mock_replica_store_zarr = Mock(spec=Store)
    replica_stores = [mock_replica_store_icechunk, mock_replica_store_zarr]

    copy_zarr_metadata(
        template_ds,
        tmp_store,
        mock_primary_icechunk,
        replica_stores=replica_stores,
        icechunk_only=True,
    )

    # Should be called for both replica and primary icechunk stores
    assert mock_copy_metadata_files.call_count == 2
    calls = mock_copy_metadata_files.call_args_list

    assert calls[0] == call(
        expected_metadata_files,
        tmp_store,
        mock_replica_store_icechunk,
    )
    assert calls[1] == call(
        expected_metadata_files,
        tmp_store,
        mock_primary_icechunk,
    )


def test_copy_zarr_metadata_noops_when_icechunk_only_and_no_icechunk_store(
    monkeypatch: pytest.MonkeyPatch,
    template_ds: xr.Dataset,
    tmp_store_and_metadata_files: tuple[Path, list[Path]],
) -> None:
    tmp_store, _expected_metadata_files = tmp_store_and_metadata_files

    mock_copy_metadata_files = Mock()
    monkeypatch.setattr(zarr_module, "_copy_metadata_files", mock_copy_metadata_files)

    mock_primary_store = Mock(spec=Store)

    copy_zarr_metadata(
        template_ds,
        tmp_store,
        mock_primary_store,
        icechunk_only=True,
    )

    assert mock_copy_metadata_files.call_count == 0


# --- sync_to_store tests ---


def test_sync_to_store_writes_bytes_to_store(tmp_path: Path) -> None:
    store_dir = tmp_path / "test.zarr"
    store = zarr.storage.LocalStore(store_dir, read_only=False)
    data = b"test-bytes"
    sync_to_store(store, "test/key", data)

    # For a LocalStore, data is stored as files on disk
    assert (store_dir / "test" / "key").read_bytes() == data


def test_sync_to_store_overwrites_existing(tmp_path: Path) -> None:
    store_dir = tmp_path / "test.zarr"
    store = zarr.storage.LocalStore(store_dir, read_only=False)
    sync_to_store(store, "key", b"original")
    sync_to_store(store, "key", b"updated")

    assert (store_dir / "key").read_bytes() == b"updated"


# --- copy_data_var tests ---


def test_copy_data_var_copies_chunks_to_primary(tmp_path: Path) -> None:
    # Build a minimal tmp store directory with chunk files
    tmp_store = tmp_path / "tmp.zarr"
    data_var_name = "temperature_2m"
    chunk_dir = tmp_store / data_var_name / "c" / "0"
    chunk_dir.mkdir(parents=True)
    chunk_file = chunk_dir / "0.0"
    chunk_file.write_bytes(b"chunk-data")

    primary_store = zarr.storage.LocalStore(tmp_path / "primary.zarr", read_only=False)

    # Minimal template_ds with encoding that matches our directory structure
    template_ds = xr.Dataset(
        {
            data_var_name: xr.Variable(
                ("time", "lat"),
                np.zeros((1, 1), dtype=np.float32),
                encoding={"shards": (1, 1), "chunks": (1, 1)},
            )
        }
    )

    copy_data_var(
        data_var_name,
        slice(0, 1),
        template_ds,
        "time",
        tmp_store,
        primary_store,
    )

    # For a LocalStore, copied chunks are stored as files on disk
    primary_store_dir = tmp_path / "primary.zarr"
    assert (primary_store_dir / data_var_name / "c" / "0" / "0.0").exists()


def test_copy_data_var_copies_to_replicas_before_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    call_order: list[str] = []

    def fake_copy_chunks(tmp_store: Path, relative_dir: str, store: Store) -> None:
        if store is mock_replica:
            call_order.append("replica")
        else:
            call_order.append("primary")

    monkeypatch.setattr(zarr_module, "_copy_data_var_chunks", fake_copy_chunks)

    tmp_store = tmp_path / "tmp.zarr"
    tmp_store.mkdir()

    mock_primary = Mock(spec=Store)
    mock_replica = Mock(spec=Store)

    template_ds = xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                ("time", "lat"),
                np.zeros((2, 2), dtype=np.float32),
                encoding={"shards": (2, 2), "chunks": (2, 2)},
            )
        }
    )

    copy_data_var(
        "temperature_2m",
        slice(0, 2),
        template_ds,
        "time",
        tmp_store,
        mock_primary,
        replica_stores=[mock_replica],
    )

    assert call_order == ["replica", "primary"]


# --- assert_fill_values_set tests ---


def test_assert_fill_values_set_passes_for_dataset_with_fill_values() -> None:
    ds = xr.Dataset(
        {"var": xr.Variable(("x",), [1.0], encoding={"fill_value": np.nan})},
        coords={"x": xr.Variable(("x",), [0], encoding={"fill_value": -1})},
    )
    assert_fill_values_set(ds)  # should not raise


def test_assert_fill_values_set_raises_for_missing_var_fill_value() -> None:
    ds = xr.Dataset(
        {"var": xr.Variable(("x",), [1.0])},  # no fill_value in encoding
        coords={"x": xr.Variable(("x",), [0], encoding={"fill_value": -1})},
    )
    with pytest.raises(AssertionError, match="var"):
        assert_fill_values_set(ds)


def test_assert_fill_values_set_raises_for_missing_coord_fill_value() -> None:
    ds = xr.Dataset(
        {"var": xr.Variable(("x",), [1.0], encoding={"fill_value": np.nan})},
        coords={"x": xr.Variable(("x",), [0])},  # no fill_value in encoding
    )
    with pytest.raises(AssertionError, match="x"):
        assert_fill_values_set(ds)


def test_assert_fill_values_set_passes_for_data_array_with_fill_value() -> None:
    da = xr.DataArray([1.0, 2.0], name="temperature")
    da.encoding["fill_value"] = np.nan
    assert_fill_values_set(da)  # should not raise


def test_assert_fill_values_set_raises_for_data_array_missing_fill_value() -> None:
    da = xr.DataArray([1.0, 2.0], name="temperature")
    with pytest.raises(AssertionError, match="temperature"):
        assert_fill_values_set(da)
