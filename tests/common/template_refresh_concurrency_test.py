from collections.abc import Sequence
from pathlib import Path

import dask.array
import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr
import zarr.storage
from zarr.abc.store import Store

from reformatters.common import template_utils
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory


def _template(size: int, *, fixed: bool = False) -> xr.DataTree:
    coords = {
        "time": ("time", np.arange(size, dtype="float64")),
        "valid_time": ("time", np.arange(size, dtype="float64") + 100),
        "runtime_state": ("time", np.full(size, np.nan)),
    }
    root = xr.Dataset(
        {"surface": ("time", dask.array.full(size, np.nan, dtype="float32"))},
        coords=coords,
        attrs={"description": "fixed" if fixed else "original"},
    )
    pressure = xr.Dataset(
        {
            "temperature": (
                ("time", "pressure_level"),
                dask.array.full((size, 2), np.nan, dtype="float32"),
            )
        },
        coords={**coords, "pressure_level": [1000.0, 850.0]},
        attrs={"description": "fixed group" if fixed else "original group"},
    )
    for ds in (root, pressure):
        for name, var in ds.variables.items():
            var.encoding.update(
                dtype=var.dtype,
                fill_value=np.nan,
                chunks=(1,) * var.ndim,
                compressors=None,
                filters=None,
            )
            if name in ds.data_vars and fixed:
                var.attrs["comment"] = "metadata fix"
                var.encoding.update(fill_value=-9999.0, _FillValue=-9999.0)
    return xr.DataTree.from_dict({"/": root, "/pressure_level": pressure})


def _write_values(repo: icechunk.Repository, size: int) -> None:
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="r+")
    for prefix in ("", "pressure_level/"):
        values = {
            "time": np.arange(size, dtype="float64"),
            "valid_time": np.arange(size, dtype="float64") + 100,
            "runtime_state": np.arange(size, dtype="float64") + 10,
        }
        if prefix:
            values["temperature"] = (
                np.arange(size * 2, dtype="float32").reshape(size, 2) + 270
            )
        else:
            values["surface"] = np.arange(size, dtype="float32") + 280
        for name, data in values.items():
            array = root[f"{prefix}{name}"]
            assert isinstance(array, zarr.Array)
            array.resize(data.shape)
            array[:] = data
    session.commit(f"Write {size} time steps")


def _repo(size: int) -> icechunk.Repository:
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    template_utils.write_metadata(
        _template(size),
        repo.writable_session("main").store,
        mode="w-",
        consolidated=False,
    )
    if size:
        _write_values(repo, size)
    return repo


def _factory(
    monkeypatch: pytest.MonkeyPatch,
    primary: icechunk.Repository,
    replicas: Sequence[icechunk.Repository | Store] = (),
) -> tuple[StoreFactory, list[icechunk.Session]]:
    sessions: list[icechunk.Session] = []

    def primary_store(
        self: StoreFactory, writable: bool = False, branch: str = "main"
    ) -> Store:
        session = (
            primary.writable_session(branch)
            if writable
            else primary.readonly_session(branch)
        )
        if writable:
            sessions.append(session)
        return session.store

    def replica_stores(
        self: StoreFactory, writable: bool = False, branch: str = "main"
    ) -> list[Store]:
        assert writable
        return [
            replica.writable_session(branch).store
            if isinstance(replica, icechunk.Repository)
            else replica
            for replica in replicas
        ]

    monkeypatch.setattr(StoreFactory, "primary_store", primary_store)
    monkeypatch.setattr(StoreFactory, "replica_stores", replica_stores)
    return (
        StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="unused", format=DatasetFormat.ICECHUNK
            ),
            dataset_id="metadata-refresh-test",
            template_config_version="v1",
        ),
        sessions,
    )


def _assert_contents(store: Store, size: int, *, fixed: bool = True) -> None:
    tree = xr.open_datatree(
        store,  # ty: ignore[invalid-argument-type]
        engine="zarr",
        chunks=None,
        consolidated=False,
    )
    assert tree.groups == ("/", "/pressure_level")
    for node in tree.subtree:
        assert node.sizes["time"] == size
        np.testing.assert_array_equal(node["runtime_state"], np.arange(size) + 10)
        np.testing.assert_array_equal(node["valid_time"], np.arange(size) + 100)
        np.testing.assert_array_equal(node["time"], np.arange(size))
    np.testing.assert_array_equal(tree["surface"], np.arange(size) + 280)
    np.testing.assert_array_equal(
        tree["pressure_level/temperature"], np.arange(size * 2).reshape(size, 2) + 270
    )
    if fixed:
        assert tree.attrs["description"] == "fixed"
        assert tree["pressure_level"].attrs["description"] == "fixed group"
        for name in ("surface", "pressure_level/temperature"):
            assert tree[name].attrs["comment"] == "metadata fix"
            assert tree[name].encoding["fill_value"] == -9999.0
            assert tree[name].encoding["_FillValue"] == -9999.0


def _append_during_validation(
    monkeypatch: pytest.MonkeyPatch,
    repo: icechunk.Repository,
    sizes: Sequence[int],
    *,
    skip: int = 0,
) -> list[int]:
    original = template_utils.assert_safe_overwrite
    pending = iter(sizes)
    appended: list[int] = []

    def validate(
        template_ds: xr.DataTree,
        existing_ds: xr.DataTree,
        append_dim: str,
        *,
        allow_new_arrays: bool,
        allow_expansion: bool,
    ) -> None:
        nonlocal skip
        original(
            template_ds,
            existing_ds,
            append_dim,
            allow_new_arrays=allow_new_arrays,
            allow_expansion=allow_expansion,
        )
        if skip:
            skip -= 1
            return
        size = next(pending, None)
        if size is not None:
            _write_values(repo, size)
            appended.append(size)

    monkeypatch.setattr(template_utils, "assert_safe_overwrite", validate)
    return appended


def test_refresh_reads_extent_from_its_writable_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(2)
    factory, sessions = _factory(monkeypatch, repo)
    primary_store = StoreFactory.primary_store
    appended: list[int] = []

    def grow_before_writable_session(
        self: StoreFactory, writable: bool = False, branch: str = "main"
    ) -> Store:
        if writable and not appended:
            _write_values(repo, 4)
            appended.append(4)
        return primary_store(self, writable=writable, branch=branch)

    monkeypatch.setattr(StoreFactory, "primary_store", grow_before_writable_session)

    template_utils.refresh_store_metadata(
        factory,
        _template(6, fixed=True),
        "time",
        tmp_path / "refresh.zarr",
        consolidated=False,
    )

    assert appended == [4]
    assert len(sessions) == 1
    _assert_contents(repo.readonly_session("main").store, 4)


def test_refresh_reopens_snapshot_after_concurrent_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(2)
    factory, sessions = _factory(monkeypatch, repo)
    appended = _append_during_validation(monkeypatch, repo, [4])

    template_utils.refresh_store_metadata(
        factory,
        _template(6, fixed=True),
        "time",
        tmp_path / "refresh.zarr",
        consolidated=False,
    )

    assert appended == [4]
    _assert_contents(repo.readonly_session("main").store, 4)
    assert len(sessions) == 2
    assert sessions[0].snapshot_id != sessions[1].snapshot_id


def test_refresh_primary_conflict_reopens_already_committed_replica_without_new_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, replica = _repo(2), _repo(4)
    factory, sessions = _factory(monkeypatch, primary, [replica])
    before = len(list(replica.ancestry(branch="main")))
    appended = _append_during_validation(monkeypatch, primary, [3], skip=1)

    template_utils.refresh_store_metadata(
        factory,
        _template(6, fixed=True),
        "time",
        tmp_path / "refresh.zarr",
        consolidated=False,
    )

    assert appended == [3]
    assert len(sessions) == 2
    assert len(list(replica.ancestry(branch="main"))) == before + 1
    _assert_contents(replica.readonly_session("main").store, 4)
    _assert_contents(primary.readonly_session("main").store, 3)


def test_refresh_conflict_retries_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(2)
    factory, sessions = _factory(monkeypatch, repo)
    appended = _append_during_validation(monkeypatch, repo, range(3, 10))

    with pytest.raises(icechunk.ConflictError):
        template_utils.refresh_store_metadata(
            factory,
            _template(10, fixed=True),
            "time",
            tmp_path / "refresh.zarr",
            consolidated=False,
        )

    assert appended == list(range(3, 8))
    assert len(sessions) == 5
    _assert_contents(repo.readonly_session("main").store, 7, fixed=False)


@pytest.mark.parametrize("replica_size", [0, 1, 4])
def test_refresh_preserves_each_replicas_extent_and_skips_unchanged_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replica_size: int
) -> None:
    primary, replica = _repo(2), _repo(replica_size)
    factory, _ = _factory(monkeypatch, primary, [replica])
    before = [len(list(repo.ancestry(branch="main"))) for repo in (primary, replica)]
    for _ in range(2):
        template_utils.refresh_store_metadata(
            factory,
            _template(6, fixed=True),
            "time",
            tmp_path / "refresh.zarr",
            consolidated=False,
        )

    for repo, size, count in zip(
        (primary, replica), (2, replica_size), before, strict=True
    ):
        assert len(list(repo.ancestry(branch="main"))) == count + bool(size)
        if size:
            _assert_contents(repo.readonly_session("main").store, size)


def test_refresh_empty_primary_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(0)
    factory, _ = _factory(monkeypatch, repo)
    before = len(list(repo.ancestry(branch="main")))
    template_utils.refresh_store_metadata(
        factory,
        _template(6, fixed=True),
        "time",
        tmp_path / "refresh.zarr",
        consolidated=False,
    )
    assert len(list(repo.ancestry(branch="main"))) == before
    assert not (tmp_path / "refresh.zarr").exists()


@pytest.mark.parametrize("consolidated", [False, True])
def test_refresh_zarr3_replica_uses_its_own_extent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consolidated: bool
) -> None:
    primary = _repo(2)
    replica = zarr.storage.MemoryStore()
    template_utils.write_metadata(
        _template(4), replica, mode="w-", consolidated=consolidated
    )
    factory, _ = _factory(monkeypatch, primary, [replica])
    template_utils.refresh_store_metadata(
        factory,
        _template(6, fixed=True),
        "time",
        tmp_path / "refresh.zarr",
        consolidated=consolidated,
    )
    tree = xr.open_datatree(
        replica,  # ty: ignore[invalid-argument-type]
        engine="zarr",
        chunks=None,
        consolidated=consolidated,
    )
    assert all(node.sizes["time"] == 4 for node in tree.subtree)
    assert tree.attrs["description"] == "fixed"
    _assert_contents(primary.readonly_session("main").store, 2)
