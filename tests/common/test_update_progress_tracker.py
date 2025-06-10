import json
from pathlib import Path

import zarr
from pytest import MonkeyPatch

from reformatters.common import zarr as common_zarr_module
from reformatters.common.update_progress_tracker import UpdateProgressTracker
from reformatters.common.zarr import get_zarr_store


def test_update_progress_tracker_close_with_local_store(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test UpdateProgressTracker loads existing progress with LocalStore (sync filesystem)"""
    monkeypatch.setattr(common_zarr_module, "_LOCAL_ZARR_STORE_BASE_PATH", tmp_path)
    store = get_zarr_store("test", "dev")
    assert not store.fs.async_impl

    # Create an existing progress file
    progress_file = (
        tmp_path
        / "test"
        / Path(store.path).name
        / "_internal_update_progress_test-job_0.json"
    )
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps({"processed_variables": ["existing_var"]}))

    tracker = UpdateProgressTracker(store, "test-job", 0)
    tracker.close()

    assert not progress_file.exists()


def test_update_progress_tracker_close_with_async_store(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test UpdateProgressTracker loads existing progress with FsspecStore (async filesystem)"""
    monkeypatch.setattr(common_zarr_module, "_LOCAL_ZARR_STORE_BASE_PATH", tmp_path)
    store = get_zarr_store("test", "dev")
    store = zarr.storage.FsspecStore.from_url(store.path)
    assert store.fs.async_impl

    # Create an existing progress file
    progress_file = (
        tmp_path
        / "test"
        / Path(store.path).name
        / "_internal_update_progress_test-job_0.json"
    )
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps({"processed_variables": ["existing_var"]}))

    tracker = UpdateProgressTracker(store, "test-job", 0)
    tracker.close()

    assert not progress_file.exists()
