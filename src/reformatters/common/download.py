from __future__ import annotations

import contextlib
import functools
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
import obstore
import requests
from obstore.exceptions import GenericError, PermissionDeniedError

from reformatters.common.logging import get_logger
from reformatters.common.retry import exponential_backoff_time

if TYPE_CHECKING:
    from obstore._store import ClientConfig, RetryConfig, S3Config
    from obstore.store import ObjectStore

log = get_logger(__name__)

DOWNLOAD_DIR = Path("data/download/")

"""
This module provides utilities for downloading files to disk over http,
with support for retries, rate limiting, and auto-coallesced byte-range requests.

The main functions which have (effectively) interchangeable interfaces are
`http_download_to_disk` which uses obstore and
`httpx_download_to_disk` which uses httpx and supports redirects and cookies.
"""


def download_to_disk(
    store: ObjectStore,
    path: str,
    local_path: Path,
    *,
    byte_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
    disk_cache: bool = False,
) -> None:
    if disk_cache and local_path.exists():
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
            file.writelines(response_buffers)

        temp_path.rename(local_path)

    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
            local_path.unlink()

        raise


_CLIENT_OPTIONS: ClientConfig = {
    "connect_timeout": "4 seconds",
    "timeout": "120 seconds",
}
_RETRY_CONFIG: RetryConfig = {
    "max_retries": 16,
    "backoff": {
        "base": 2,
        "init_backoff": timedelta(seconds=1),
        "max_backoff": timedelta(seconds=16),
    },
    # A backstop, shouldn't hit this with the above backoff settings
    "retry_timeout": timedelta(minutes=5),
}


@functools.cache
def http_store(base_url: str) -> obstore.store.HTTPStore:
    """
    A obstore.store.HTTPStore tuned to maximize chance of success at the expense
    of latency, while not waiting indefinitely for unresponsive servers.
    """
    return obstore.store.HTTPStore.from_url(
        base_url,
        client_options={**_CLIENT_OPTIONS, "user_agent": "dynamical.org reformatters"},
        retry_config=_RETRY_CONFIG,
    )


@functools.cache
def s3_store(
    bucket_url: str,
    region: str,
    skip_signature: bool = True,
    endpoint: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    virtual_hosted_style_request: bool | None = None,
) -> obstore.store.S3Store:
    """An S3 store, anonymous by default. Pass credentials and `endpoint` for a signed
    store on an S3-compatible service such as R2."""
    config: S3Config = {}
    if endpoint is not None:
        config["endpoint"] = endpoint
    if access_key_id is not None:
        config["access_key_id"] = access_key_id
    if secret_access_key is not None:
        config["secret_access_key"] = secret_access_key
    if virtual_hosted_style_request is not None:
        config["virtual_hosted_style_request"] = virtual_hosted_style_request
    store = obstore.store.from_url(
        bucket_url,
        config=config,
        region=region,
        skip_signature=skip_signature,
        client_options=_CLIENT_OPTIONS,
        retry_config=_RETRY_CONFIG,
    )
    assert isinstance(store, obstore.store.S3Store)
    return store


@functools.cache
def gcs_store(bucket_url: str) -> obstore.store.GCSStore:
    """An anonymous obstore store for a public gs://bucket."""
    store = obstore.store.from_url(
        bucket_url,
        skip_signature=True,
        client_options=_CLIENT_OPTIONS,
        retry_config=_RETRY_CONFIG,
    )
    assert isinstance(store, obstore.store.GCSStore)
    return store


def http_download_to_disk(
    url: str,
    dataset_id: str,
    byte_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
    local_path_suffix: str = "",
    disk_cache: bool = False,
) -> Path:
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    store = http_store(base_url)
    local_path = get_local_path(dataset_id, parsed_url.path, local_path_suffix)
    download_to_disk(
        store,
        parsed_url.path,
        local_path,
        disk_cache=disk_cache,
        byte_ranges=byte_ranges,
    )
    return local_path


def s3_download_to_disk(
    url: str,
    dataset_id: str,
    *,
    region: str,
    byte_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
    local_path_suffix: str = "",
    disk_cache: bool = False,
) -> Path:
    """Download s3://bucket/key to disk via the cached anonymous obstore S3 store."""
    parsed_url = urlparse(url)
    assert parsed_url.scheme == "s3", url
    store = s3_store(f"s3://{parsed_url.netloc}", region=region)
    local_path = get_local_path(dataset_id, parsed_url.path, local_path_suffix)
    download_to_disk(
        store,
        parsed_url.path.removeprefix("/"),
        local_path,
        disk_cache=disk_cache,
        byte_ranges=byte_ranges,
    )
    return local_path


def gcs_download_to_disk(url: str, dataset_id: str) -> Path:
    """Download gs://bucket/key to disk via the cached anonymous obstore GCS store."""
    parsed_url = urlparse(url)
    assert parsed_url.scheme == "gs", url
    local_path = get_local_path(dataset_id, parsed_url.path)
    download_to_disk(
        gcs_store(f"gs://{parsed_url.netloc}"),
        parsed_url.path.removeprefix("/"),
        local_path,
    )
    return local_path


def s3_read_bytes(url: str, *, region: str, start: int, end: int) -> bytes:
    """The `[start, end)` byte range of s3://bucket/key, via the anonymous S3 store."""
    parsed_url = urlparse(url)
    assert parsed_url.scheme == "s3", url
    store = s3_store(f"s3://{parsed_url.netloc}", region=region)
    return bytes(
        obstore.get_range(
            store, parsed_url.path.removeprefix("/"), start=start, end=end
        )
    )


def get_local_path(dataset_id: str, path: str, local_path_suffix: str = "") -> Path:
    base_local = DOWNLOAD_DIR / dataset_id / path.removeprefix("/")
    return (
        base_local.with_name(base_local.name + local_path_suffix)
        if local_path_suffix
        else base_local
    )


@functools.cache
def _httpx_client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=4.0, read=120.0, write=120.0, pool=120.0),
        headers={"User-Agent": "dynamical.org reformatters"},
    )


class RateLimiter:
    """Token bucket rate limiter. Allows bursting up to `burst` requests, then enforces max_per_minute."""

    def __init__(self, max_per_minute: int, burst: int = 1) -> None:
        self._rate = max_per_minute / 60.0  # tokens/second
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._burst, self._tokens + (now - self._last) * self._rate
            )
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
            else:
                sleep_time = (1.0 - self._tokens) / self._rate
                time.sleep(sleep_time)
                self._tokens = 0.0
                self._last = time.monotonic()


_DEFAULT_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 16
_RETRY_TIMEOUT_SECONDS = 300.0


def httpx_get(
    url: str,
    headers: dict[str, str] | None = None,
    rate_limiter: RateLimiter | None = None,
    retry_status_codes: set[int] = _DEFAULT_RETRY_STATUS_CODES,
) -> httpx.Response:
    def read(response: httpx.Response) -> httpx.Response:
        response.read()
        return response

    return _httpx_stream_with_retries(
        url, read, headers, rate_limiter, retry_status_codes
    )


def _httpx_stream_with_retries[T](
    url: str,
    consume: Callable[[httpx.Response], T],
    headers: dict[str, str] | None,
    rate_limiter: RateLimiter | None,
    retry_status_codes: set[int],
) -> T:
    """GET `url` and pass the streaming response to `consume`, retrying the whole
    request when it fails in transport, including a body cut off mid-stream."""
    client = _httpx_client()
    start_time = time.monotonic()

    last_exception: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if time.monotonic() - start_time > _RETRY_TIMEOUT_SECONDS:
            break

        if attempt > 0:
            time.sleep(exponential_backoff_time(attempt - 1))

        if rate_limiter is not None:
            rate_limiter.wait()

        try:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code not in retry_status_codes:
                    response.raise_for_status()
                    return consume(response)
        except httpx.TransportError as e:
            last_exception = e
            log.warning(f"httpx transport error on attempt {attempt + 1}: {e}")
            continue

        last_exception = httpx.HTTPStatusError(
            f"Server returned {response.status_code}",
            request=response.request,
            response=response,
        )
        log.warning(
            f"Retryable status {response.status_code} on attempt {attempt + 1} for {url}"
        )

    assert last_exception is not None
    raise last_exception


def _parse_multipart_byteranges(data: bytes, content_type: str) -> bytes:
    # Extract boundary from Content-Type: multipart/byteranges; boundary=XXXX
    _, _, params = content_type.partition(";")
    boundary = b""
    for raw_param in params.split(";"):
        stripped = raw_param.strip()
        if stripped.startswith("boundary="):
            boundary = stripped.removeprefix("boundary=").strip().encode()
            break
    assert boundary, f"No boundary found in content_type: {content_type}"

    delimiter = b"--" + boundary
    parts = data.split(delimiter)
    # First part is empty (before first delimiter), last part is "--\r\n" (closing)
    body_parts: list[bytes] = []
    for part in parts[1:]:
        if part.strip() == b"--" or not part.strip():
            continue
        # Each part has headers separated from body by \r\n\r\n
        header_end = part.find(b"\r\n\r\n")
        assert header_end != -1, "Malformed multipart part: no header/body separator"
        body = part[header_end + 4 :]
        # Strip trailing \r\n that precedes the next delimiter
        if body.endswith(b"\r\n"):
            body = body[:-2]
        body_parts.append(body)

    assert body_parts, "No parts found in multipart response"
    return b"".join(body_parts)


def httpx_download_to_disk(
    url: str,
    dataset_id: str,
    byte_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
    local_path_suffix: str = "",
    rate_limiter: RateLimiter | None = None,
    retry_status_codes: set[int] = _DEFAULT_RETRY_STATUS_CODES,
    disk_cache: bool = False,
) -> Path:
    """httpx based download which supports redirects and maintains cookies."""
    parsed_url = urlparse(url)
    local_path = get_local_path(dataset_id, parsed_url.path, local_path_suffix)
    if disk_cache and local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_name(f"{local_path.name}.{uuid.uuid4().hex[:8]}")

    try:
        if byte_ranges is not None:
            starts, ends = byte_ranges
            # Build multi-range header. Ends from grib index are exclusive; HTTP Range is inclusive.
            range_specs = [f"{s}-{e - 1}" for s, e in zip(starts, ends, strict=True)]
            range_header = f"bytes={', '.join(range_specs)}"
            response = httpx_get(
                url,
                headers={"Range": range_header},
                rate_limiter=rate_limiter,
                retry_status_codes=retry_status_codes,
            )

            content_type = response.headers.get("content-type", "")
            if "multipart/byteranges" in content_type:
                body = _parse_multipart_byteranges(response.content, content_type)
            else:
                # Single range or server returned full content
                body = response.content

            with open(temp_path, "wb") as f:
                f.write(body)
        else:

            def write(response: httpx.Response) -> None:
                with open(temp_path, "wb") as f:
                    f.writelines(response.iter_bytes())

            _httpx_stream_with_retries(
                url, write, None, rate_limiter, retry_status_codes
            )

        temp_path.rename(local_path)

    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
            local_path.unlink()
        raise

    return local_path


def http_status_code(e: Exception) -> int | None:
    """HTTP status code from an httpx or requests error response, else None."""
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code
    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
        return e.response.status_code
    return None


# obstore raises PermissionDeniedError for a 403 and GenericError for everything it doesn't
# map to a specific type, including both missing data and retry-exhausted transient errors.
FALLBACK_EXCEPTIONS = (FileNotFoundError, GenericError, PermissionDeniedError)

# A range request against a zero size object gets a 416 whose body reports the object's real
# size. A 416 against a non-empty object means we asked for the wrong bytes, which is a bug.
_EMPTY_OBJECT_RANGE_ERROR = "<ActualObjectSize>0</ActualObjectSize>"


def is_not_found(e: Exception) -> bool:
    """Whether e indicates the source file has no data, as opposed to an unexpected failure."""
    if isinstance(e, FileNotFoundError | PermissionDeniedError):
        return True
    if isinstance(e, GenericError):
        return _EMPTY_OBJECT_RANGE_ERROR in str(e)
    return http_status_code(e) in (HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND)
