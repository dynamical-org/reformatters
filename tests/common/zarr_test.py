from pathlib import Path
from unittest.mock import Mock

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
def template_ds() -> xr.DataTree:
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                coords={
                    "time": ["2000-01-01", "2000-01-02"],
                    "lat": [10, 20, 30],
                    "lon": [100, 110, 120],
                }
            )
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
    template_ds: xr.DataTree,
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
        files, _store_path, store, _skip = call_args[0]
        # Verify the files list contains the expected files (order-independent)
        assert set(files) == set(expected_metadata_files)
        _assert_format_specific_order(files, store)


def _assert_format_specific_order(files: list[Path], store: Store) -> None:
    """Zarr v3 readers need coordinate chunks written before metadata; a fresh
    icechunk store rejects a chunk whose array metadata doesn't exist yet."""
    last_json = max(i for i, f in enumerate(files) if f.name == "zarr.json")
    first_chunk = min(i for i, f in enumerate(files) if f.name != "zarr.json")
    if isinstance(store, IcechunkStore):
        assert last_json < first_chunk
    else:
        assert first_chunk < last_json


def test_copy_zarr_metadata_skips_non_icechunk_stores_when_icechunk_only(
    monkeypatch: pytest.MonkeyPatch,
    template_ds: xr.DataTree,
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

    for call_args, expected_store in zip(
        calls, [mock_replica_store_icechunk, mock_primary_icechunk], strict=True
    ):
        files, store_path, store, _skip = call_args[0]
        assert set(files) == set(expected_metadata_files)
        _assert_format_specific_order(files, store)
        assert store_path == tmp_store
        assert store is expected_store


def test_copy_zarr_metadata_noops_when_icechunk_only_and_no_icechunk_store(
    monkeypatch: pytest.MonkeyPatch,
    template_ds: xr.DataTree,
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


@pytest.fixture
def multi_group_template_ds() -> xr.DataTree:
    root = xr.Dataset(
        coords={"time": ["2000-01-01", "2000-01-02"], "lat": [10, 20, 30]}
    )
    pressure = xr.Dataset(
        coords={
            "time": ["2000-01-01", "2000-01-02"],
            "lat": [10, 20, 30],
            "pressure_level": [1000.0, 850.0],
        }
    )
    return xr.DataTree.from_dict({"/": root, "/pressure_level": pressure})


@pytest.fixture
def multi_group_store(tmp_path: Path) -> Path:
    """A tmp store with a root node and a pressure_level child group, each carrying
    coord chunks and a zarr.json (mirrors what to_zarr writes for a DataTree)."""
    store_path = tmp_path / "tmp_store"

    (store_path / "zarr.json").parent.mkdir(parents=True)
    (store_path / "zarr.json").touch()
    for coord in ("time", "lat"):
        chunk_dir = store_path / coord / "c"
        chunk_dir.mkdir(parents=True)
        (chunk_dir / "0").touch()
        (store_path / coord / "zarr.json").touch()

    group = store_path / "pressure_level"
    (group / "zarr.json").parent.mkdir(parents=True)
    (group / "zarr.json").touch()
    for coord in ("time", "lat", "pressure_level"):
        chunk_dir = group / coord / "c"
        chunk_dir.mkdir(parents=True)
        (chunk_dir / "0").touch()
        (group / coord / "zarr.json").touch()

    return store_path


def test_coord_chunk_globs_collects_group_nested_coord_chunk(
    monkeypatch: pytest.MonkeyPatch,
    multi_group_template_ds: xr.DataTree,
    multi_group_store: Path,
) -> None:
    captured: dict[str, list[Path]] = {}

    def fake_copy_metadata_files(
        files: list[Path], _store_path: Path, _store: Store, _skip: bool = False
    ) -> None:
        captured["files"] = files

    monkeypatch.setattr(zarr_module, "_copy_metadata_files", fake_copy_metadata_files)

    copy_zarr_metadata(multi_group_template_ds, multi_group_store, Mock(spec=Store))

    relative = {f.relative_to(multi_group_store).as_posix() for f in captured["files"]}
    # The group's own vertical coord chunk lives under the group prefix and must be collected.
    assert "pressure_level/pressure_level/c/0" in relative


def test_coord_chunk_globs_collects_scalar_coord_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A scalar coord (e.g. spatial_ref) stores its single chunk as the file
    # `spatial_ref/c`, not `spatial_ref/c/0`; the `c/**/*` glob alone would miss it.
    template = xr.DataTree.from_dict(
        {"/": xr.Dataset(coords={"lat": [10, 20], "spatial_ref": ((), np.array(0))})}
    )
    store = tmp_path / "store"
    (store / "zarr.json").parent.mkdir(parents=True)
    (store / "zarr.json").touch()
    (store / "lat" / "c").mkdir(parents=True)
    (store / "lat" / "c" / "0").touch()
    (store / "spatial_ref").mkdir()
    (store / "spatial_ref" / "c").touch()  # scalar chunk is a file named "c"

    captured: dict[str, list[Path]] = {}
    monkeypatch.setattr(
        zarr_module,
        "_copy_metadata_files",
        lambda files, _store_path, _store, _skip=False: captured.__setitem__(
            "files", files
        ),
    )
    copy_zarr_metadata(template, store, Mock(spec=Store))

    relative = {f.relative_to(store).as_posix() for f in captured["files"]}
    assert "spatial_ref/c" in relative  # the scalar chunk file is collected
    assert "lat/c/0" in relative  # and chunked coords still are


def test_copy_zarr_metadata_icechunk_writes_root_zarr_json_before_group(
    monkeypatch: pytest.MonkeyPatch,
    multi_group_template_ds: xr.DataTree,
    multi_group_store: Path,
) -> None:
    captured: dict[str, list[Path]] = {}

    def fake_copy_metadata_files(
        files: list[Path], _store_path: Path, _store: Store, _skip: bool = False
    ) -> None:
        captured["files"] = files

    monkeypatch.setattr(zarr_module, "_copy_metadata_files", fake_copy_metadata_files)

    copy_zarr_metadata(
        multi_group_template_ds, multi_group_store, Mock(spec=IcechunkStore)
    )

    order = [f.relative_to(multi_group_store).as_posix() for f in captured["files"]]
    # Shallowest-first: a parent group's metadata is written before its child's, so
    # icechunk never rejects a child array whose parent group does not yet exist.
    assert order.index("zarr.json") < order.index("pressure_level/zarr.json")


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
