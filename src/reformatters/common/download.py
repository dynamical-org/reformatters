import contextlib
import functools
import os
import uuid
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import obstore

DOWNLOAD_DIR = Path("data/download/")


def download_to_disk(
    store: obstore.store.HTTPStore,
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


def http_download(
    url: str, filename: str, subdirectory: str, overwrite_existing: bool = False
) -> Path:
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    store = http_store(base_url)
    local_path = DOWNLOAD_DIR / subdirectory / filename
    download_to_disk(
        store, parsed_url.path, local_path, overwrite_existing=overwrite_existing
    )
    return local_path
