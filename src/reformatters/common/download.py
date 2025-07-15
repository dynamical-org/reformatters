from __future__ import annotations

import contextlib
import functools
import os
import uuid
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import obstore

if TYPE_CHECKING:
    from obstore.store import ObjectStore

DOWNLOAD_DIR = Path("data/download/")


def download_to_disk(
    store: ObjectStore,
    path: str,
    local_path: Path,
    *,
    byte_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
    overwrite_existing: bool,
) -> None:
    if not overwrite_existing and local_path.exists():
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    response_buffers: obstore.BytesStream | list[obstore.Bytes]
    if byte_ranges is not None:
        byte_range_starts, byte_range_ends = byte_ranges[0], byte_ranges[1]
        response_buffers = obstore.get_ranges(
            store=store, path=path, starts=byte_range_starts, ends=byte_range_ends
        )
    else:
        response_buffers = obstore.get(store, path).stream()

    # Avoid race by writing to temp file then renaming when complete
    temp_path = local_path.with_name(f"{local_path.name}.{uuid.uuid4().hex[:8]}")

    try:
        with open(temp_path, "wb") as file:
            for buffer in response_buffers:
                file.write(buffer)

        temp_path.rename(local_path)

    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temp_path)
            os.remove(local_path)

        raise


@functools.cache
def http_store(base_url: str) -> obstore.store.HTTPStore:
    """
    A obstore.store.HTTPStore tuned to maximize chance of success at the expense
    of latency, while not waiting indefinitely for unresponsive servers.
    """
    return obstore.store.HTTPStore.from_url(
        base_url,
        client_options={
            "connect_timeout": "4 seconds",
            "timeout": "16 seconds",
        },
        retry_config={
            "max_retries": 16,
            "backoff": {
                "base": 2,
                "init_backoff": timedelta(seconds=1),
                "max_backoff": timedelta(seconds=16),
            },
            # A backstop, shouldn't hit this with the above backoff settings
            "retry_timeout": timedelta(minutes=5),
        },
    )


@functools.cache
def s3_store(
    bucket_url: str, region: str, skip_signature: bool = True
) -> obstore.store.S3Store:
    store = obstore.store.from_url(
        bucket_url,
        region=region,
        skip_signature=skip_signature,
        client_options={
            "connect_timeout": "4 seconds",
            "timeout": "120 seconds",
        },
        retry_config={
            "max_retries": 16,
            "backoff": {
                "base": 2,
                "init_backoff": timedelta(seconds=1),
                "max_backoff": timedelta(seconds=16),
            },
            # A backstop, shouldn't hit this with the above backoff settings
            "retry_timeout": timedelta(minutes=5),
        },
    )
    assert isinstance(store, obstore.store.S3Store)
    return store


def http_download_to_disk(
    url: str,
    dataset_id: str,
    byte_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
    local_path_suffix: str = "",
) -> Path:
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    store = http_store(base_url)
    local_path = get_local_path(dataset_id, parsed_url.path, local_path_suffix)
    download_to_disk(
        store,
        parsed_url.path,
        local_path,
        overwrite_existing=True,
        byte_ranges=byte_ranges,
    )
    return local_path


def get_local_path(dataset_id: str, path: str, local_path_suffix: str = "") -> Path:
    base_local = DOWNLOAD_DIR / dataset_id / path.removeprefix("/")
    return (
        base_local.with_name(base_local.name + local_path_suffix)
        if local_path_suffix
        else base_local
    )
