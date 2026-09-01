"""Real and synthetic NOAA GRIB index files for tests of virtual region jobs."""

import itertools
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from reformatters.common.download import http_download_to_disk

_S3_HTTPS_PREFIX = "https://{bucket}.s3.amazonaws.com/"
_copy_counter = itertools.count()


def cached_grib_index(url: str, dataset_id: str) -> Path:
    """The shared on-disk download of a real NOAA `.idx`.

    Never hand this path to code that deletes the index it was given; pass it through
    `stub_grib_index_download`, which copies it per call.
    """
    if url.startswith("s3://"):
        bucket, _, key = url.removeprefix("s3://").partition("/")
        url = _S3_HTTPS_PREFIX.format(bucket=bucket) + key
    return http_download_to_disk(url, dataset_id, disk_cache=True)


def stub_grib_index_download(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    tmp_path: Path,
    make_index: Callable[[str], str | Path],
) -> None:
    """Point `module.s3_download_to_disk` at a fresh index file per call.

    `make_index` maps a source URL to either the index text to write or the path of a
    real index to copy; either way each call yields a new file under `tmp_path`.

    A virtual region job's `file_refs` deletes the index it downloads, in a `finally`
    block, and that is correct: the index is a temporary download and cleaning it up is
    that function's job. The trap is on the test side. A test that writes its `.idx`
    fixture once at the download path has that fixture consumed by the first call and is
    meaningless on every run after, while still reporting green -- which is the state CI
    and a reviewer see. Fix it here by handing over a fresh copy per call; never by
    making the production path skip the unlink.
    """

    def download(url: str, dataset_id: str, **kwargs: object) -> Path:
        index = make_index(url)
        path = tmp_path / f"{next(_copy_counter)}-{url.rsplit('/', 1)[-1]}"
        if isinstance(index, Path):
            path.write_bytes(index.read_bytes())
        else:
            path.write_text(index)
        return path

    monkeypatch.setattr(module, "s3_download_to_disk", download)
