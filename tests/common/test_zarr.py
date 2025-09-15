from pathlib import Path
from unittest.mock import Mock, call

import pytest
import xarray as xr
import zarr
from icechunk.store import IcechunkStore

from reformatters.common import zarr as zarr_module
from reformatters.common.zarr import copy_zarr_metadata


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
    # Simulate template_ds.coords - we know it's ["time", "lat", "lon"] from the test
    for coord in ["time", "lat", "lon"]:
        metadata_files.extend(
            f for f in store_path.glob(f"{coord}/c/**/*") if f.is_file()
        )
    metadata_files.append(store_path / "zarr.json")
    metadata_files.extend(
        store_path.glob("*/zarr.json")
    )  # This will be in filesystem order

    return store_path, metadata_files


def test_copy_zarr_metadata_calls_copy_metadata_files_for_all_stores(
    monkeypatch: pytest.MonkeyPatch,
    template_ds: xr.Dataset,
    tmp_store_and_metadata_files: tuple[Path, list[Path]],
) -> None:
    tmp_store, expected_metadata_files = tmp_store_and_metadata_files

    mock_copy_metadata_files = Mock()
    monkeypatch.setattr(zarr_module, "_copy_metadata_files", mock_copy_metadata_files)

    mock_primary_store = Mock(spec=zarr.abc.store.Store)
    mock_replica_store_zarr = Mock(spec=zarr.abc.store.Store)
    mock_replica_store_icechunk = Mock(spec=IcechunkStore)
    replica_stores = [mock_replica_store_zarr, mock_replica_store_icechunk]

    copy_zarr_metadata(template_ds, tmp_store, mock_primary_store, replica_stores)

    # Should be called once for each replica store, then once for primary store
    assert mock_copy_metadata_files.call_count == 3
    calls = mock_copy_metadata_files.call_args_list

    # Instead of exact call matching, check the calls contain the right elements
    assert len(calls) == 3
    for call_args in calls:
        files, store_path, store = call_args[0]
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
    mock_replica_store_zarr = Mock(spec=zarr.abc.store.Store)
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
    tmp_store, expected_metadata_files = tmp_store_and_metadata_files

    mock_copy_metadata_files = Mock()
    monkeypatch.setattr(zarr_module, "_copy_metadata_files", mock_copy_metadata_files)

    mock_primary_store = Mock(spec=zarr.abc.store.Store)

    copy_zarr_metadata(
        template_ds,
        tmp_store,
        mock_primary_store,
        icechunk_only=True,
    )

    assert mock_copy_metadata_files.call_count == 0
