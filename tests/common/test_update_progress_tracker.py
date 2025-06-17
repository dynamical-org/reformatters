import json
from pathlib import Path

import zarr

from reformatters.common.update_progress_tracker import UpdateProgressTracker
from reformatters.common.zarr import get_zarr_store


def test_update_progress_tracker_close_with_local_store(tmp_path: Path) -> None:
    """Test UpdateProgressTracker loads existing progress with LocalStore (sync filesystem)"""
    store = get_zarr_store("fake-prod-path", "test", "dev")
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


def test_update_progress_tracker_close_with_async_store(tmp_path: Path) -> None:
    """Test UpdateProgressTracker loads existing progress with FsspecStore (async filesystem)"""
    store = get_zarr_store("fake-prod-path", "test", "dev")
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
