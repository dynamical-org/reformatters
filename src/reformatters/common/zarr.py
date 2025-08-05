from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

import fsspec  # type: ignore
import xarray as xr
import zarr
from fsspec.implementations.local import LocalFileSystem  # type: ignore

from reformatters.common.config import Config
from reformatters.common.logging import get_logger

logger = get_logger(__name__)

_LOCAL_ZARR_STORE_BASE_PATH = "data/output"

BLOSC_2BYTE_ZSTD_LEVEL3_SHUFFLE = zarr.codecs.BloscCodec(
    typesize=2,
    cname="zstd",
    clevel=3,
    shuffle="shuffle",  # byte shuffle to improve compression
).to_dict()


BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE = zarr.codecs.BloscCodec(
    typesize=4,
    cname="zstd",
    clevel=3,
    shuffle="shuffle",  # byte shuffle to improve compression
).to_dict()

BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE = zarr.codecs.BloscCodec(
    typesize=8,
    cname="zstd",
    clevel=3,
    shuffle="shuffle",
).to_dict()


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
            f"{_LOCAL_ZARR_STORE_BASE_PATH}/{dataset_id}/v{version}.zarr"
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


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/{uuid4()}-tmp.zarr").absolute()


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


def copy_data_var(
    data_var_name: str,
    i_slice: slice,
    template_ds: xr.Dataset,
    append_dim: str,
    tmp_store: Path,
    primary_store: zarr.abc.store.Store,
    track_progress_callback: Callable[[], None] | None = None,
) -> None:
    dim_index = template_ds[data_var_name].dims.index(append_dim)
    append_dim_shard_size = template_ds[data_var_name].encoding["shards"][dim_index]
    shard_index = i_slice.start // append_dim_shard_size
    assert dim_index == 0  # relative_dir format below assumes append dim is first
    relative_dir = f"{data_var_name}/c/{shard_index}/"

    logger.info(
        f"Copying data var chunks to primary store ({primary_store}) for {relative_dir}."
    )

    for file in tmp_store.glob(f"{relative_dir}**/*"):
        if not file.is_file():
            continue
        key = str(file.relative_to(tmp_store))
        sync_to_store(primary_store, key, file.read_bytes())

    if track_progress_callback is not None:
        track_progress_callback()

    try:
        # Delete data to conserve disk space.
        for file in tmp_store.glob(f"{relative_dir}**/*"):
            if file.is_file():
                file.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete chunk after upload: {e}")


def sync_to_store(store: zarr.abc.store.Store, key: str, data: bytes) -> None:
    zarr.core.sync.sync(
        store.set(
            key,
            # Should we use zarr.core.buffer.default_buffer_prototype().buffer.from_bytes( ?
            zarr.core.buffer.cpu.Buffer.from_bytes(data),
        )
    )


def copy_zarr_metadata(
    template_ds: xr.Dataset, tmp_store: Path, primary_store: zarr.abc.store.Store
) -> None:
    logger.info(f"Copying metadata to primary store ({primary_store}) from {tmp_store}")

    metadata_files: list[Path] = []

    # The coordinate label arrays must be copied before the metadata.
    for coord in template_ds.coords:
        metadata_files.extend(
            f for f in tmp_store.glob(f"{coord}/c/**/*") if f.is_file()
        )

    metadata_files.append(tmp_store / "zarr.json")
    metadata_files.extend(tmp_store.glob("*/zarr.json"))

    for file in metadata_files:
        relative_path = str(file.relative_to(tmp_store))
        sync_to_store(primary_store, relative_path, file.read_bytes())


def _get_fs_and_path(
    store: zarr.abc.store.Store,
) -> tuple[fsspec.AbstractFileSystem, str]:
    """Gross work around to allow us to make other store types quack like FsspecStore."""
    fs = getattr(store, "fs", None)
    if not isinstance(fs, fsspec.AbstractFileSystem):
        raise ValueError(
            "primary_store must have an fs that is an instance of fsspec.AbstractFileSystem"
        )
    path = getattr(store, "path", None)
    if not isinstance(path, str):
        raise ValueError("primary_store must have a path attribute that is a string")
    return fs, path
