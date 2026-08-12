import itertools
import json
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
import zarr
from icechunk import IcechunkStore
from zarr.abc.store import RangeByteRequest, Store, SuffixByteRequest
from zarr.codecs import ShardingCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync

from reformatters.common.logging import get_logger
from reformatters.common.retry import retry
from scripts.validation.scan_common import find_registered_dataset
from scripts.validation.utils import open_readonly_store, resolve_output_dir

log = get_logger(__name__)

_MAX_UINT64 = np.uint64((1 << 64) - 1)
_READ_BATCH_SIZE = 256
_READ_PARALLELISM = 8
_variables_option = typer.Option(
    None, "--variable", "-v", help="Float variable to scan; repeat as needed"
)
_output_dir_option = typer.Option(
    None, "--output-dir", help="Directory for chunk_census.json"
)


@dataclass(frozen=True)
class ArrayChunkCensus:
    expected_shards: int
    present_shards: int
    expected_chunks: int
    present_chunks: int

    @property
    def absent_shards(self) -> int:
        return self.expected_shards - self.present_shards

    @property
    def absent_chunks(self) -> int:
        return self.expected_chunks - self.present_chunks


def _batched[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _get_partial_values(
    store: Store,
    requests: Sequence[tuple[str, RangeByteRequest | SuffixByteRequest]],
) -> list[Any | None]:
    prototype = default_buffer_prototype()
    return retry(lambda: sync(store.get_partial_values(prototype, requests)))


def _get_partial_value_batches(
    store: Store,
    requests: Sequence[tuple[str, RangeByteRequest | SuffixByteRequest]],
) -> list[list[Any | None]]:
    batches = _batched(requests, _READ_BATCH_SIZE)
    with ThreadPoolExecutor(
        max_workers=min(_READ_PARALLELISM, len(batches))
    ) as executor:
        return list(
            executor.map(lambda batch: _get_partial_values(store, batch), batches)
        )


def _expected_local_chunk_slices(
    shard_coords: tuple[int, ...],
    chunks_per_shard: tuple[int, ...],
    chunk_counts: tuple[int, ...],
) -> tuple[slice, ...]:
    return tuple(
        slice(0, min(per_shard, total - shard * per_shard))
        for shard, per_shard, total in zip(
            shard_coords, chunks_per_shard, chunk_counts, strict=True
        )
    )


def _index_size(codec: ShardingCodec, chunks_per_shard: tuple[int, ...]) -> int:
    index_codec_names = [item.to_dict()["name"] for item in codec.index_codecs]
    assert index_codec_names == ["bytes", "crc32c"], (
        f"Unsupported sharding index codecs: {index_codec_names}"
    )
    return math.prod(chunks_per_shard) * 2 * np.dtype("<u8").itemsize + 4


def census_array_chunks(store: Store, path: str) -> ArrayChunkCensus:
    array = zarr.open_array(store=store, path=path, mode="r")
    metadata = array.metadata
    assert isinstance(metadata, ArrayV3Metadata)
    shape = tuple(array.shape)
    chunks = tuple(metadata.chunks)
    shards = metadata.shards
    chunk_counts = tuple(
        math.ceil(size / chunk) for size, chunk in zip(shape, chunks, strict=True)
    )

    if shards is None:
        chunk_coords = list(
            itertools.product(*(range(count) for count in chunk_counts))
        )
        requests = [
            (
                f"{path}/{metadata.encode_chunk_key(coords)}",
                RangeByteRequest(0, 1),
            )
            for coords in chunk_coords
        ]
        present = sum(
            value is not None
            for batch in _get_partial_value_batches(store, requests)
            for value in batch
        )
        return ArrayChunkCensus(
            expected_shards=len(requests),
            present_shards=present,
            expected_chunks=len(requests),
            present_chunks=present,
        )

    shards = tuple(shards)
    assert all(shard % chunk == 0 for shard, chunk in zip(shards, chunks, strict=True))
    chunks_per_shard = tuple(
        shard // chunk for shard, chunk in zip(shards, chunks, strict=True)
    )
    shard_counts = tuple(
        math.ceil(size / shard) for size, shard in zip(shape, shards, strict=True)
    )
    shard_coords = list(itertools.product(*(range(count) for count in shard_counts)))
    codec = metadata.codecs[0]
    assert isinstance(codec, ShardingCodec)
    assert codec.index_location == "end"
    index_size = _index_size(codec, chunks_per_shard)
    requests = [
        (
            f"{path}/{metadata.encode_chunk_key(coords)}",
            SuffixByteRequest(index_size),
        )
        for coords in shard_coords
    ]

    expected_chunks = math.prod(chunk_counts)
    present_shards = 0
    present_chunks = 0
    for coord_batch, value_batch in zip(
        _batched(shard_coords, _READ_BATCH_SIZE),
        _get_partial_value_batches(store, requests),
        strict=True,
    ):
        for coords, value in zip(
            coord_batch,
            value_batch,
            strict=True,
        ):
            if value is None:
                continue
            present_shards += 1
            index_bytes = value.to_bytes()
            assert len(index_bytes) == index_size
            offsets_and_lengths = np.frombuffer(index_bytes[:-4], dtype="<u8").reshape(
                (*chunks_per_shard, 2)
            )
            expected_slice = _expected_local_chunk_slices(
                coords, chunks_per_shard, chunk_counts
            )
            present_chunks += int(
                np.count_nonzero(
                    offsets_and_lengths[expected_slice][..., 0] != _MAX_UINT64
                )
            )

    return ArrayChunkCensus(
        expected_shards=math.prod(shard_counts),
        present_shards=present_shards,
        expected_chunks=expected_chunks,
        present_chunks=present_chunks,
    )


def chunk_census(
    dataset_url: str,
    variables: list[str] | None = _variables_option,
    output_dir: Path | None = _output_dir_option,
) -> None:
    """Exhaustively count physically present chunks, including chunks inside shards."""
    store_like, consolidated = open_readonly_store(dataset_url)
    assert isinstance(store_like, Store), (
        f"Chunk census needs a Zarr Store: {store_like}"
    )
    root = zarr.open_group(store=store_like, mode="r", use_consolidated=consolidated)
    dataset_id = str(root.attrs["dataset_id"])
    dataset = find_registered_dataset(dataset_id)
    assert dataset is not None, f"Dataset {dataset_id!r} is not registered"
    float_paths = [
        var.path
        for var in dataset.template_config.data_vars
        if np.issubdtype(np.dtype(var.encoding.dtype), np.floating)
    ]
    selected = variables or float_paths
    unknown = set(selected) - set(float_paths)
    assert not unknown, f"Unknown float variables: {sorted(unknown)}"

    results: dict[str, dict[str, int]] = {}
    for path in selected:
        result = census_array_chunks(store_like, path)
        results[path] = asdict(result) | {
            "absent_shards": result.absent_shards,
            "absent_chunks": result.absent_chunks,
        }
        log.info(
            f"{path}: {result.present_chunks}/{result.expected_chunks} chunks, "
            f"{result.present_shards}/{result.expected_shards} shards present"
        )

    out = resolve_output_dir(dataset_url, output_dir)
    report = {
        "dataset_id": dataset_id,
        "dataset_url": dataset_url,
        "icechunk_snapshot_id": (
            str(store_like.session.snapshot_id)
            if isinstance(store_like, IcechunkStore)
            else None
        ),
        "scanned_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "variables": results,
    }
    report_path = out / "chunk_census.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log.info(f"Wrote {report_path}")
