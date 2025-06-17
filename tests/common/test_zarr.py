from pathlib import Path

import zarr
from pytest import MonkeyPatch

from reformatters.common.config import Config, Env
from reformatters.common.zarr import get_mode, get_zarr_store


def test_get_zarr_store_dev(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "env", Env.dev)
    store = get_zarr_store("fake-prod-path", "test-dataset", "1.0.0")
    assert isinstance(store, zarr.storage.LocalStore)
    assert store.path == str(Path("data/output/test-dataset/vdev.zarr").absolute())


def test_get_zarr_store_prod(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "env", Env.prod)
    store = get_zarr_store("s3://fake-prod-path", "test-dataset", "1.0.0")
    assert isinstance(store, zarr.storage.FsspecStore)
    assert store.path == "fake-prod-path/test-dataset/v1.0.0.zarr"


def test_get_mode() -> None:
    # Dev and temp stores can be overwritten, everything else should not overwrite.
    dev_store = zarr.storage.LocalStore(Path("test-dev.zarr"))
    assert get_mode(dev_store) == "w"

    tmp_store = Path("test-tmp.zarr")
    assert get_mode(tmp_store) == "w"

    prod_store = zarr.storage.FsspecStore.from_url("s3://test-bucket/test.zarr")
    assert get_mode(prod_store) == "w-"
