"""Real and synthetic NOAA GRIB index files for tests of virtual region jobs."""

import itertools
import struct
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from reformatters.common.download import http_download_to_disk
from reformatters.noaa.noaa_virtual_region_job import GRIB_SECTION_0_BYTES

_S3_HTTPS_PREFIX = "https://{bucket}.s3.amazonaws.com/"
_copy_counter = itertools.count()


def cached_grib_index(url: str, dataset_id: str) -> Path:
    """The shared on-disk download of a real NOAA `.idx`.

    Never hand this path to code that deletes the index it was given; pass it through
    `stub_grib_source_file_reads`, which copies it per call.
    """
    if url.startswith("s3://"):
        bucket, _, key = url.removeprefix("s3://").partition("/")
        url = _S3_HTTPS_PREFIX.format(bucket=bucket) + key
    return http_download_to_disk(url, dataset_id, disk_cache=True)


def stub_grib_source_file_reads(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    tmp_path: Path,
    make_index: Callable[[str], str | Path],
    *,
    data_file_size: int | None = None,
) -> None:
    """Stand in for the source file reads `file_refs` makes: the index download from
    `module.s3_download_to_disk` and the GRIB header read from `module.s3_read_bytes`.

    `make_index` maps a source URL to either the index text to write or the path of a
    real index to copy; either way each call yields a new file under `tmp_path`.

    A fresh copy per call is the contract: `file_refs` deletes the index it is given, so
    a fixture written once at the download path is consumed by the first call and every
    later call in the same run sees nothing.

    The synthesized GRIB header agrees with the index, so `file_refs` reaches its
    message matching; a test of the stale-index guard stubs `s3_read_bytes` itself.
    `data_file_size` supplies the first message's length for a single-message index,
    whose end byte is the end of the data file rather than the next message's start.
    """

    def download(url: str, dataset_id: str, **kwargs: object) -> Path:
        index = make_index(url + ".idx")
        path = tmp_path / f"{next(_copy_counter)}-{url.rsplit('/', 1)[-1]}"
        if isinstance(index, Path):
            path.write_bytes(index.read_bytes())
        else:
            path.write_text(index)
        return path

    def read_bytes(url: str, **kwargs: object) -> bytes:
        # Called with the data file's url, while make_index is keyed on the index's.
        index = make_index(url + ".idx")
        text = index.read_text() if isinstance(index, Path) else index
        starts = [int(line.split(":")[1]) for line in text.splitlines() if line]
        if len(starts) > 1:
            length = starts[1] - starts[0]
        else:
            assert data_file_size is not None, (
                "A single-message index needs data_file_size to imply a message length"
            )
            length = data_file_size - starts[0]
        return grib_section_0(length)

    monkeypatch.setattr(module, "s3_download_to_disk", download)
    monkeypatch.setattr(module, "s3_read_bytes", read_bytes)


def grib_section_0(message_length: int) -> bytes:
    """A GRIB2 section 0 declaring `message_length`, as the data file's first bytes."""
    header = b"GRIB" + bytes([0, 0, 0, 2]) + struct.pack(">Q", message_length)
    assert len(header) == GRIB_SECTION_0_BYTES
    return header
