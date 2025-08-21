from pathlib import Path
from typing import Literal

import zarr
from fsspec.implementations.local import LocalFileSystem  # type: ignore

from reformatters.common import storage
from reformatters.common.config import Config


def get_zarr_store(
    prod_base_path: str, dataset_id: str, version: str
) -> zarr.storage.FsspecStore:
    if not Config.is_prod:
        version = "dev" if Config.is_dev else version

        # This should work, but it gives FileNotFoundError when it should be creating a new zarr.
        # return zarr.storage.FsspecStore.from_url(
        #     "file://data/output/noaa/gefs/forecast/dev.zarr"
        # )
        # Instead make a zarr LocalStore and attach an fsspec filesystem to it.
        # Technically that filesystem should be an AsyncFileSystem to match
        # zarr.storage.FsspecStore but AsyncFileSystem does not support _put.
        local_path = Path(
            f"{storage._LOCAL_ZARR_STORE_BASE_PATH}/{dataset_id}/v{version}.zarr"
        ).absolute()

        store = zarr.storage.LocalStore(local_path)

        fs = LocalFileSystem(auto_mkdir=True)
        store.fs = fs  # type: ignore[attr-defined]
        store.path = str(local_path)  # type: ignore[attr-defined]

        # We are duck typing this LocalStore as a FsspecStore
        return store  # type: ignore[return-value]

    return zarr.storage.FsspecStore.from_url(
        f"{prod_base_path}/{dataset_id}/v{version}.zarr"
    )


def get_mode(
    store: zarr.abc.store.Store | Path,
) -> Literal["w-", "w"]:
    if isinstance(store, zarr.storage.FsspecStore):
        path_str = store.path
    elif isinstance(store, zarr.storage.LocalStore):
        path_str = store.root.name
    elif isinstance(store, Path):
        path_str = store.name
    else:
        raise ValueError(f"Unexpected store type: {type(store)}")

    if path_str.endswith(("dev.zarr", "-tmp.zarr")):
        return "w"  # Allow overwritting dev store

    return "w-"  # Safe default - don't overwrite
