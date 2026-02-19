from collections.abc import Callable, Iterable
from pathlib import Path

import xarray as xr
import zarr
import zarr.buffer
import zarr.core.sync
from icechunk.store import IcechunkStore
from zarr.abc.store import Store
from zarr.codecs import BloscCodec

from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

log = get_logger(__name__)

_LOCAL_ZARR_STORE_BASE_PATH = "data/output"

BLOSC_2BYTE_ZSTD_LEVEL3_SHUFFLE = BloscCodec(
    typesize=2,
    cname="zstd",
    clevel=3,
    shuffle="shuffle",  # byte shuffle to improve compression
).to_dict()


BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE = BloscCodec(
    typesize=4,
    cname="zstd",
    clevel=3,
    shuffle="shuffle",  # byte shuffle to improve compression
).to_dict()

BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE = BloscCodec(
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
    primary_store: Store,
    replica_stores: Iterable[Store] = (),
    track_progress_callback: Callable[[], None] | None = None,
) -> None:
    dim_index = template_ds[data_var_name].dims.index(append_dim)
    append_dim_shard_size = template_ds[data_var_name].encoding["shards"][dim_index]
    shard_index = i_slice.start // append_dim_shard_size
    assert dim_index == 0  # relative_dir format below assumes append dim is first
    relative_dir = f"{data_var_name}/c/{shard_index}/"

    for replica_store in replica_stores:
        log.info(
            f"Copying data var chunks to replica store ({replica_store}) for {relative_dir}."
        )
        _copy_data_var_chunks(tmp_store, relative_dir, replica_store)
        log.info(
            f"Done copying data var chunks to replica store ({replica_store}) for {relative_dir}."
        )

    log.info(
        f"Copying data var chunks to primary store ({primary_store}) for {relative_dir}."
    )
    _copy_data_var_chunks(tmp_store, relative_dir, primary_store)
    log.info(
        f"Done copying data var chunks to primary store ({primary_store}) for {relative_dir}."
    )

    if track_progress_callback is not None:
        track_progress_callback()

    try:
        # Delete data to free disk space.
        for file in tmp_store.glob(f"{relative_dir}**/*"):
            if file.is_file():
                file.unlink()
    except Exception as e:  # noqa: BLE001
        log.warning(f"Failed to delete chunk after upload: {e}")


def _copy_data_var_chunks(
    tmp_store: Path,
    relative_dir: str,
    store: Store,
) -> None:
    for file in tmp_store.glob(f"{relative_dir}**/*"):
        if not file.is_file():
            continue
        key = file.relative_to(tmp_store).as_posix()
        sync_to_store(store, key, file.read_bytes())


def copy_zarr_metadata(
    template_ds: xr.Dataset,
    tmp_store: Path,
    primary_store: Store,
    replica_stores: Iterable[Store] = (),
    icechunk_only: bool = False,
) -> None:
    """
    Copy the metadata and coordinate label arrays from the temporary store to the primary and replica stores.

    In the zarr3 case, the updated metadata will become available to readers, which is why we update the metadata
    after we have already written the actual data chunks.

    In the Icechunk case, we need to update the metadata before writing data chunks, as Icechunk will throw an error
    if we try to write data that does not match the shape specified in the metadata. This is safe to do so, however,
    because in the Icechunk case, data is not available to readers until we commit the Icechunk writable session.
    """
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
        if icechunk_only and not isinstance(replica_store, IcechunkStore):
            log.info(
                f"Skipping metadata copy to replica store ({replica_store}) because it is not an IcechunkStore and icechunk_only is True"
            )
            continue

        log.info(
            f"Copying metadata to replica store ({replica_store}) from {tmp_store}"
        )
        _copy_metadata_files(metadata_files, tmp_store, replica_store)

    if icechunk_only and not isinstance(primary_store, IcechunkStore):
        log.info(
            f"Skipping metadata copy to primary store ({primary_store}) because it is not an IcechunkStore and icechunk_only is True"
        )
        return

    log.info(f"Copying metadata to primary store ({primary_store}) from {tmp_store}")
    _copy_metadata_files(metadata_files, tmp_store, primary_store)


def _copy_metadata_files(
    metadata_files: list[Path],
    tmp_store: Path,
    store: Store,
) -> None:
    for file in metadata_files:
        relative_path = file.relative_to(tmp_store).as_posix()
        sync_to_store(store, relative_path, file.read_bytes())


def sync_to_store(store: Store, key: str, data: bytes) -> None:
    retry(
        lambda: zarr.core.sync.sync(
            store.set(
                key,
                zarr.buffer.default_buffer_prototype().buffer.from_bytes(data),
            ),
            timeout=90,  # In seconds. Timeout needs to be long enough to upload a large shard.
        ),
        max_attempts=6,
    )


def assert_fill_values_set(xr_obj: xr.Dataset | xr.DataArray) -> None:
    if isinstance(xr_obj, xr.DataArray):
        assert "fill_value" in xr_obj.encoding, (
            f"Fill value not set for DataArray {xr_obj.name}"
        )

    elif isinstance(xr_obj, xr.Dataset):
        for coord_name, coord in xr_obj.coords.items():
            assert "fill_value" in coord.encoding, (
                f"Fill value not set for coordinate {coord_name}"
            )
        for var_name, var in xr_obj.data_vars.items():
            assert "fill_value" in var.encoding, (
                f"Fill value not set for variable {var_name}"
            )
    else:
        raise ValueError(f"Expected xr.Dataset or xr.DataArray, got {type(xr_obj)}")
