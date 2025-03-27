import time
from functools import cache
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Literal
from uuid import uuid4

import fsspec  # type: ignore
import numpy as np
import xarray as xr
import zarr
from fsspec.implementations.local import LocalFileSystem  # type: ignore

from reformatters.common.config import Config
from reformatters.common.logging import get_logger
from reformatters.common.types import ArrayFloat32

logger = get_logger(__name__)

_LOCAL_ZARR_STORE_BASE_PATH = "data/output"


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


def get_zarr_store(dataset_id: str, version: str) -> zarr.storage.FsspecStore:
    if not Config.is_prod:
        version = "dev"

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
        f"s3://us-west-2.opendata.source.coop/dynamical/{dataset_id}/v{version}.zarr",
        storage_options={
            "key": Config.source_coop.aws_access_key_id,
            "secret": Config.source_coop.aws_secret_access_key,
        },
    )


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/{uuid4()}-tmp.zarr").absolute()


def get_mode(
    store: zarr.storage.FsspecStore | zarr.storage.LocalStore | Path,
) -> Literal["w-", "w"]:
    if isinstance(store, zarr.storage.FsspecStore):
        path_str = store.path
    elif isinstance(store, zarr.storage.LocalStore):
        path_str = store.root.name
    elif isinstance(store, Path):
        path_str = store.name
    else:
        raise ValueError(f"Unexpected store type: {type(store)}")

    if path_str.endswith("dev.zarr") or path_str.endswith("-tmp.zarr"):
        return "w"  # Allow overwritting dev store

    return "w-"  # Safe default - don't overwrite


def copy__name(
    data_var_name: str,
    chunk_index: int,
    tmp_store: Path,
    final_store: zarr.storage.FsspecStore,
) -> None:
    relative_dir = f"{data_var_name}/c/{chunk_index}/"

    logger.info(
        f"Copying data var chunks to final store ({final_store}) for {relative_dir}."
    )
    fs = final_store.fs
    fs.auto_mkdir = True

    # We want to support local and s3fs filesystems. fsspec local filesystem is sync,
    # but our s3fs from zarr.storage.FsspecStore is async and here we work around it.
    # The AsyncFileSystem wrapper on LocalFilesystem raises NotImplementedError when _put is called.
    source = f"{tmp_store / relative_dir}/"
    dest = f"{final_store.path}/{relative_dir}"
    _copy_to_store("put", source, dest, fs, recursive=True, auto_mkdir=True)

    try:
        # Delete data to conserve disk space.
        for file in tmp_store.glob(f"{relative_dir}**/*"):
            if file.is_file():
                file.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete chunk after upload: {e}")


def copy_zarr_metadata(
    template_ds: xr.Dataset, tmp_store: Path, final_store: zarr.storage.FsspecStore
) -> None:
    logger.info(f"Copying metadata to final store ({final_store}) from {tmp_store}")

    metadata_files: list[Path] = []

    # The coordinate label arrays must be copied before the metadata.
    for coord in template_ds.coords:
        metadata_files.extend(
            f for f in tmp_store.glob(f"{coord}/c/**/*") if f.is_file()
        )

    metadata_files.append(tmp_store / "zarr.json")
    metadata_files.extend(tmp_store.glob("*/zarr.json"))

    # This could be partially parallelized BUT make sure to write the coords before the metadata.
    for file in metadata_files:
        relative = file.relative_to(tmp_store)
        dest = f"{final_store.path}/{relative}"
        _copy_to_store("put_file", file, dest, final_store.fs)


def _copy_to_store(
    method: Literal["put", "put_file"],
    source: str | Path,
    dest: str,
    dest_fs: fsspec.AbstractFileSystem,
    **kwargs: bool,
) -> None:
    """
    Copy a file or directory to the store's filesystem.

    This function handles both sync and async filesystems. The fsspec local filesystem is sync,
    but the fsspec store from zarr.storage.FsspecStore is async, so we need to handle both cases.
    (The AsyncFileSystem wrapper on LocalFilesystem raises NotImplementedError when _put is called
    so we can't just use that.)
    """
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            if hasattr(dest_fs, "_put"):
                # Zarr's FsspecStore creates async fsspec filesystems, so use their sync method
                zarr.core.sync.sync(
                    getattr(dest_fs, "_" + method)(source, dest, **kwargs)
                )
            else:
                getattr(dest_fs, method)(source, dest, **kwargs)
            break
        except Exception:
            if attempt == max_attempts - 1:  # Last attempt failed
                raise
            time.sleep(1)
            continue


def create_data_array_and_template(
    chunk_template_ds: xr.Dataset,
    data_var_name: str,
    shared_buffer: SharedMemory,
) -> tuple[xr.DataArray, xr.DataArray]:
    # This template is small and we will pass it between processes
    data_array_template = chunk_template_ds[data_var_name]

    # This data array will be assigned actual, shared memory
    data_array = data_array_template.copy()

    # Drop all non-dimension coordinates from the template only,
    # they are already written by write_metadata.
    data_array_template = data_array_template.drop_vars(
        [
            coord
            for coord in data_array_template.coords
            if coord not in data_array_template.dims
        ]
    )

    shared_array: ArrayFloat32 = np.ndarray(
        data_array.shape,
        dtype=data_array.dtype,
        buffer=shared_buffer.buf,
    )
    # Important:
    # We rely on initializing with nans so failed reads (eg. corrupt source data)
    # leave nan and to reuse the same shared buffer for each variable.
    shared_array[:] = np.nan
    data_array.data = shared_array

    return data_array, data_array_template
