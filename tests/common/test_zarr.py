from pathlib import Path

import zarr

from reformatters.common.config import Config, Env
from reformatters.common.zarr import get_mode, get_zarr_store


def test_get_zarr_store_dev() -> None:
    # Test dev environment store creation
    Config.env = Env.dev
    store = get_zarr_store("test-dataset", "1.0.0")
    assert isinstance(store, zarr.storage.LocalStore)
    assert store.path == str(Path("data/output/test-dataset/vdev.zarr").absolute())


def test_get_zarr_store_prod() -> None:
    # Test prod environment store creation
    Config.env = Env.prod
    store = get_zarr_store("test-dataset", "1.0.0")
    assert isinstance(store, zarr.storage.FsspecStore)
    assert (
        store.path
        == "us-west-2.opendata.source.coop/dynamical/test-dataset/v1.0.0.zarr"
    )


def test_get_mode() -> None:
    # Test mode selection for different store types
    dev_store = zarr.storage.LocalStore(Path("test-dev.zarr"))
    assert get_mode(dev_store) == "w"

    prod_store = zarr.storage.FsspecStore.from_url("s3://test-bucket/test.zarr")
    assert get_mode(prod_store) == "w-"

    tmp_store = Path("test-tmp.zarr")
    assert get_mode(tmp_store) == "w"
