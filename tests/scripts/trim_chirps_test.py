from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from scripts.trim_chirps import DATASET_IDS, main, trim_store


def _create_store(path: Path, last: int | None = 6) -> icechunk.Repository:
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    data = np.full((10, 4, 4), np.nan, dtype=np.float32)
    if last is not None:
        data[:last, :, :] = 0
        data[last, -1, -1] = 0
        if last > 2:
            data[2, :, :] = np.nan
    ds = xr.Dataset(
        {"precipitation_surface": (("time", "latitude", "longitude"), data)},
        coords={
            "time": pd.date_range("2025-01-01", periods=10),
            "latitude": np.arange(4),
            "longitude": np.arange(4),
            "quality": ("time", np.arange(10)),
        },
        attrs={"dataset_id": DATASET_IDS[0], "description": "Preserve metadata"},
    )
    ds.to_zarr(
        session.store,
        zarr_format=3,
        consolidated=False,
        encoding={"precipitation_surface": {"chunks": (2, 2, 2), "shards": (4, 4, 4)}},
    )
    session.commit("fixture")
    return repo


@pytest.mark.parametrize("last", [0, 3, 6, 9])
def test_trim_preserves_values_metadata_and_old_snapshot(
    tmp_path: Path, last: int
) -> None:
    repo = _create_store(tmp_path / "store", last)
    before_snapshot = repo.lookup_branch("main")
    before = xr.open_zarr(repo.readonly_session("main").store, chunks=None).load()
    dry_run = trim_store(repo, DATASET_IDS[0])
    assert dry_run.after_size == last + 1
    assert dry_run.after_end == pd.Timestamp("2025-01-01") + pd.Timedelta(days=last)
    assert dry_run.after_snapshot is None
    assert repo.lookup_branch("main") == before_snapshot

    committed = trim_store(repo, DATASET_IDS[0], commit=True)
    assert (committed.after_snapshot is not None) == (last < 9)
    for consolidated in (False, None):
        after = xr.open_zarr(
            repo.readonly_session("main").store, chunks=None, consolidated=consolidated
        )
        xr.testing.assert_identical(after, before.isel(time=slice(0, last + 1)))
    old = xr.open_zarr(
        repo.readonly_session(snapshot_id=before_snapshot).store, chunks=None
    )
    xr.testing.assert_identical(old, before)
    old_root = zarr.open_group(
        repo.readonly_session(snapshot_id=before_snapshot).store, mode="r"
    )
    new_root = zarr.open_group(repo.readonly_session("main").store, mode="r")
    for name, array in old_root.arrays():
        expected = array.metadata.to_dict()
        if name in ("time", "quality", "precipitation_surface"):
            expected["shape"] = (last + 1, *array.shape[1:])
        assert new_root[name].metadata.to_dict() == expected
    assert trim_store(repo, DATASET_IDS[0], commit=True).after_snapshot is None


def test_last_day_is_union_of_data_variables(tmp_path: Path) -> None:
    repo = _create_store(tmp_path / "store", 3)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, use_consolidated=False)
    values = np.full(10, np.nan, dtype=np.float32)
    values[7] = 0
    root.create_array(
        "other", data=values, chunks=(2,), dimension_names=("time",), fill_value=np.nan
    )
    session.commit("second variable")
    assert trim_store(repo, DATASET_IDS[0], commit=True).after_size == 8
    after = xr.open_zarr(repo.readonly_session("main").store, chunks=None)
    assert after.other.size == after.precipitation_surface.sizes["time"] == 8


def test_rejects_empty_or_wrong_dataset(tmp_path: Path) -> None:
    repo = _create_store(tmp_path / "store", None)
    before = repo.lookup_branch("main")
    with pytest.raises(AssertionError, match="refusing to empty"):
        trim_store(repo, DATASET_IDS[0], commit=True)
    with pytest.raises(AssertionError):
        trim_store(repo, DATASET_IDS[1], commit=True)
    assert repo.lookup_branch("main") == before


@pytest.mark.parametrize("during_commit", [False, True])
def test_concurrent_update_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_commit: bool
) -> None:
    repo = _create_store(tmp_path / "store")
    writable_session = repo.writable_session

    def advance_main() -> None:
        other = writable_session("main")
        zarr.open_group(other.store).attrs["concurrent_update"] = True
        other.commit("concurrent update")

    if during_commit:
        resize = zarr.Array.resize
        advanced = False

        def racing_resize(array: zarr.Array, shape: tuple[int, ...]) -> None:
            nonlocal advanced
            resize(array, shape)
            if not advanced:
                advance_main()
                advanced = True

        monkeypatch.setattr(zarr.Array, "resize", racing_resize)
        expected_error = icechunk.ConflictError
    else:

        def racing_session(branch: str) -> icechunk.Session:
            advance_main()
            return writable_session(branch)

        monkeypatch.setattr(repo, "writable_session", racing_session)
        expected_error = AssertionError
    with pytest.raises(expected_error):
        trim_store(repo, DATASET_IDS[0], commit=True)
    after = xr.open_zarr(repo.readonly_session("main").store, chunks=None)
    assert after.sizes["time"] == 10
    assert after.attrs["concurrent_update"]


def test_cli_defaults_to_dry_run(tmp_path: Path) -> None:
    path = tmp_path / "store"
    repo = _create_store(path)
    before = repo.lookup_branch("main")
    main([DATASET_IDS[0], str(path)])
    assert repo.lookup_branch("main") == before
    main([DATASET_IDS[0], str(path), "--commit"])
    assert repo.lookup_branch("main") != before
