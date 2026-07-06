from collections.abc import Iterable
from pathlib import Path

import xarray as xr
import zarr
import zarr.buffer
import zarr.core.sync
from icechunk.store import IcechunkStore
from zarr.abc.store import Store
from zarr.codecs import BloscCodec

from reformatters.common.iterating import node_path_prefix
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

log = get_logger(__name__)

_LOCAL_ZARR_STORE_BASE_PATH = "data/output"


def _store_repr(store: Store) -> str:
    return str(store).replace("\n", " ")


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
) -> None:
    dim_index = template_ds[data_var_name].dims.index(append_dim)
    append_dim_shard_size = template_ds[data_var_name].encoding["shards"][dim_index]
    shard_index = i_slice.start // append_dim_shard_size
    assert dim_index == 0  # relative_dir format below assumes append dim is first
    relative_dir = f"{data_var_name}/c/{shard_index}/"

    for replica_store in replica_stores:
        log.info(
            f"Copying data var chunks to replica store ({_store_repr(replica_store)}) for {relative_dir}."
        )
        _copy_data_var_chunks(tmp_store, relative_dir, replica_store)
        log.info(
            f"Done copying data var chunks to replica store ({_store_repr(replica_store)}) for {relative_dir}."
        )

    log.info(
        f"Copying data var chunks to primary store ({_store_repr(primary_store)}) for {relative_dir}."
    )
    _copy_data_var_chunks(tmp_store, relative_dir, primary_store)
    log.info(
        f"Done copying data var chunks to primary store ({_store_repr(primary_store)}) for {relative_dir}."
    )

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
        key = str(file.relative_to(tmp_store))
        sync_to_store(store, key, file.read_bytes())


def _coord_chunk_globs(template_ds: xr.DataTree) -> list[str]:
    """Glob patterns for every coordinate's chunk files, group-prefixed. The `c/**/*`
    pattern catches chunked coords (`<coord>/c/0`, `<coord>/c/0/0`); the bare `c` catches
    a scalar coord whose single chunk is the file `<coord>/c` (the caller's is_file filter
    drops the `c/` directory matched by the bare pattern for non-scalar coords)."""
    globs = []
    for node in template_ds.subtree:
        prefix = node_path_prefix(node)
        for coord in node.to_dataset().coords:
            globs.append(f"{prefix}{coord}/c/**/*")
            globs.append(f"{prefix}{coord}/c")
    return globs


def copy_zarr_metadata(
    template_ds: xr.DataTree,
    tmp_store: Path,
    primary_store: Store,
    replica_stores: Iterable[Store] = (),
    icechunk_only: bool = False,
    zarr3_only: bool = False,
    skip_unchanged: bool = False,
) -> None:
    """
    Copy the metadata and coordinate label arrays from the temporary store to the primary and replica stores.

    In the zarr3 case, the updated metadata will become available to readers, which is why we update the metadata
    after we have already written the actual data chunks.

    In the Icechunk case, we need to update the metadata before writing data chunks, as Icechunk will throw an error
    if we try to write data that does not match the shape specified in the metadata. This is safe to do so, however,
    because in the Icechunk case, data is not available to readers until we commit the Icechunk writable session.
    """
    assert not (icechunk_only and zarr3_only)

    coord_chunk_files: list[Path] = []
    for coord_glob in _coord_chunk_globs(template_ds):
        coord_chunk_files.extend(f for f in tmp_store.glob(coord_glob) if f.is_file())
    # Shallowest first so a parent group's metadata is written before its children's
    # (icechunk rejects a child array whose parent group does not yet exist).
    zarr_json_files = sorted(
        tmp_store.rglob("zarr.json"),
        key=lambda p: len(p.relative_to(tmp_store).parts),
    )

    def _ordered_files(store: Store) -> list[Path]:
        # Zarr v3: coordinate label chunks before metadata, so readers never see
        # metadata referencing coordinate values that aren't written yet.
        # Icechunk: metadata first — a fresh store rejects a chunk whose array
        # doesn't exist yet — and the session commit is atomic for readers.
        if isinstance(store, IcechunkStore):
            return [*zarr_json_files, *coord_chunk_files]
        return [*coord_chunk_files, *zarr_json_files]

    def _should_skip(store: Store) -> bool:
        is_icechunk = isinstance(store, IcechunkStore)
        return (icechunk_only and not is_icechunk) or (zarr3_only and is_icechunk)

    # It is important that we copy metadata to replica stores first
    # Since the primary store is our reference store (to determine what data we have and what needs to be written)
    # we only want to update its metadata once we are sure the replicas are up to date.
    for replica_store in replica_stores:
        if _should_skip(replica_store):
            continue

        log.info(
            f"Copying metadata to replica store ({_store_repr(replica_store)}) from {tmp_store}"
        )
        _copy_metadata_files(
            _ordered_files(replica_store), tmp_store, replica_store, skip_unchanged
        )

    if _should_skip(primary_store):
        return

    log.info(
        f"Copying metadata to primary store ({_store_repr(primary_store)}) from {tmp_store}"
    )
    _copy_metadata_files(
        _ordered_files(primary_store), tmp_store, primary_store, skip_unchanged
    )


def _copy_metadata_files(
    metadata_files: list[Path],
    tmp_store: Path,
    store: Store,
    skip_unchanged: bool = False,
) -> None:
    for file in metadata_files:
        relative_path = str(file.relative_to(tmp_store))
        data = file.read_bytes()
        # skip_unchanged keeps a metadata refresh from dirtying the session (and so
        # committing) when the store already matches the template byte-for-byte.
        if skip_unchanged and _store_bytes_equal(store, relative_path, data):
            continue
        sync_to_store(store, relative_path, data)


def _store_bytes_equal(store: Store, key: str, data: bytes) -> bool:
    existing = zarr.core.sync.sync(
        store.get(key, prototype=zarr.buffer.default_buffer_prototype())
    )
    return existing is not None and existing.to_bytes() == data


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


def assert_fill_values_set(
    xr_obj: xr.Dataset | xr.DataArray | xr.DataTree,
) -> None:
    if isinstance(xr_obj, xr.DataTree):
        for node in xr_obj.subtree:
            assert_fill_values_set(node.to_dataset())
        return

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
