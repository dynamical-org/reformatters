from collections.abc import Callable, Iterable
from pathlib import Path

import xarray as xr
import zarr

from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

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


def copy_data_var(
    data_var_name: str,
    i_slice: slice,
    template_ds: xr.Dataset,
    append_dim: str,
    tmp_store: Path,
    primary_store: zarr.abc.store.Store,
    replica_stores: Iterable[zarr.abc.store.Store] = (),
    track_progress_callback: Callable[[], None] | None = None,
) -> None:
    dim_index = template_ds[data_var_name].dims.index(append_dim)
    append_dim_shard_size = template_ds[data_var_name].encoding["shards"][dim_index]
    shard_index = i_slice.start // append_dim_shard_size
    assert dim_index == 0  # relative_dir format below assumes append dim is first
    relative_dir = f"{data_var_name}/c/{shard_index}/"

    for replica_store in replica_stores:
        logger.info(
            f"Copying data var chunks to replica store ({replica_store}) for {relative_dir}."
        )
        _copy_data_var_chunks(tmp_store, relative_dir, replica_store)
        logger.info(
            f"Done copying data var chunks to replica store ({replica_store}) for {relative_dir}."
        )

    logger.info(
        f"Copying data var chunks to primary store ({primary_store}) for {relative_dir}."
    )
    _copy_data_var_chunks(tmp_store, relative_dir, primary_store)
    logger.info(
        f"Done copying data var chunks to primary store ({primary_store}) for {relative_dir}."
    )

    if track_progress_callback is not None:
        track_progress_callback()

    try:
        # Delete data to free disk space.
        for file in tmp_store.glob(f"{relative_dir}**/*"):
            if file.is_file():
                file.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete chunk after upload: {e}")


def _copy_data_var_chunks(
    tmp_store: Path,
    relative_dir: str,
    store: zarr.abc.store.Store,
) -> None:
    for file in tmp_store.glob(f"{relative_dir}**/*"):
        if not file.is_file():
            continue
        key = str(file.relative_to(tmp_store))
        sync_to_store(store, key, file.read_bytes())


def copy_zarr_metadata(
    template_ds: xr.Dataset,
    tmp_store: Path,
    primary_store: zarr.abc.store.Store,
    replica_stores: Iterable[zarr.abc.store.Store] = (),
) -> None:
    metadata_files: list[Path] = []

    # The coordinate label arrays must be copied before the metadata.
    for coord in template_ds.coords:
        metadata_files.extend(
            f for f in tmp_store.glob(f"{coord}/c/**/*") if f.is_file()
        )

    metadata_files.append(tmp_store / "zarr.json")
    metadata_files.extend(tmp_store.glob("*/zarr.json"))

    # It is important that we copy metadata to replica stores first
    # Since the primary store is our reference store (to determine what data we have and what needs to be written)
    # we only want to update its metadata once we are sure the replicas are up to date.
    for replica_store in replica_stores:
        logger.info(
            f"Copying metadata to replica store ({replica_store}) from {tmp_store}"
        )
        _copy_metadata_files(metadata_files, tmp_store, replica_store)

    logger.info(f"Copying metadata to primary store ({primary_store}) from {tmp_store}")
    _copy_metadata_files(metadata_files, tmp_store, primary_store)


def _copy_metadata_files(
    metadata_files: list[Path],
    tmp_store: Path,
    store: zarr.abc.store.Store,
) -> None:
    for file in metadata_files:
        relative_path = str(file.relative_to(tmp_store))
        sync_to_store(store, relative_path, file.read_bytes())


def sync_to_store(store: zarr.abc.store.Store, key: str, data: bytes) -> None:
    retry(
        lambda: zarr.core.sync.sync(
            store.set(
                key,
                zarr.core.buffer.default_buffer_prototype().buffer.from_bytes(data),
            ),
            timeout=90,  # In seconds. Timeout needs to be long enough to upload a large shard.
        ),
        max_attempts=6,
    )
