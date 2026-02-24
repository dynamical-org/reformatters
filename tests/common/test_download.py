from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock, patch

import obstore.store
import pytest

from reformatters.common import download as download_module
from reformatters.common.download import (
    DOWNLOAD_DIR,
    download_to_disk,
    get_local_path,
    http_download_to_disk,
    http_store,
    s3_store,
)


def test_get_local_path_basic() -> None:
    result = get_local_path("my-dataset", "/some/data/file.grib2")
    assert result == DOWNLOAD_DIR / "my-dataset" / "some/data/file.grib2"


def test_get_local_path_strips_leading_slash() -> None:
    with_slash = get_local_path("ds", "/a/b.txt")
    without_slash = get_local_path("ds", "a/b.txt")
    assert with_slash == without_slash


def test_get_local_path_with_suffix() -> None:
    result = get_local_path("ds", "/data/file.grib2", local_path_suffix="-abc123")
    assert result.name == "file.grib2-abc123"
    base = get_local_path("ds", "/data/file.grib2")
    assert result.parent == base.parent


def test_get_local_path_no_suffix_returns_base() -> None:
    result = get_local_path("ds", "/data/file.grib2", local_path_suffix="")
    assert result == DOWNLOAD_DIR / "ds" / "data/file.grib2"


def test_http_store_returns_http_store() -> None:
    store = http_store("https://example.com")
    assert isinstance(store, obstore.store.HTTPStore)


def test_http_store_is_cached() -> None:
    store1 = http_store("https://example.com/path")
    store2 = http_store("https://example.com/path")
    assert store1 is store2


def test_s3_store_returns_s3_store() -> None:
    store = s3_store("s3://noaa-gefs-pds", region="us-east-1", skip_signature=True)
    assert isinstance(store, obstore.store.S3Store)


def test_download_to_disk_skips_if_exists_and_no_overwrite(tmp_path: Path) -> None:
    local_path = tmp_path / "existing_file.bin"
    local_path.write_bytes(b"existing content")

    mock_store = MagicMock()
    download_to_disk(mock_store, "some/path", local_path, overwrite_existing=False)

    mock_store.get.assert_not_called()
    mock_store.get_ranges.assert_not_called()


def test_download_to_disk_writes_file(tmp_path: Path) -> None:
    local_path = tmp_path / "subdir" / "file.bin"
    file_content = b"hello world"

    mock_get_result = MagicMock()
    mock_get_result.stream.return_value = iter([file_content])
    mock_store = MagicMock()

    with (
        patch("obstore.get", return_value=mock_get_result),
        patch.object(mock_get_result, "stream", return_value=iter([file_content])),
    ):
        download_to_disk(mock_store, "some/path", local_path, overwrite_existing=True)

    assert local_path.exists()
    assert local_path.read_bytes() == file_content


def test_download_to_disk_cleans_up_temp_on_error(tmp_path: Path) -> None:
    local_path = tmp_path / "file.bin"
    mock_store = MagicMock()

    with (
        patch("obstore.get", side_effect=OSError("network error")),
        pytest.raises(OSError, match="network error"),
    ):
        download_to_disk(mock_store, "some/path", local_path, overwrite_existing=True)

    assert not local_path.exists()
    temp_files = list(tmp_path.glob("*.???????"))
    assert len(temp_files) == 0


def test_http_download_to_disk_calls_download(tmp_path: Path) -> None:
    captured: list[dict] = []

    def fake_download_to_disk(
        store: object,
        path: str,
        local_path: Path,
        *,
        byte_ranges: tuple[Sequence[int], Sequence[int]] | None,
        overwrite_existing: bool,
    ) -> None:
        captured.append(
            {
                "path": path,
                "local_path": local_path,
                "byte_ranges": byte_ranges,
                "overwrite_existing": overwrite_existing,
            }
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"data")

    with patch.object(download_module, "download_to_disk", fake_download_to_disk):
        result = http_download_to_disk(
            "https://example.com/data/file.grib2", "my-dataset"
        )

    assert len(captured) == 1
    assert captured[0]["path"] == "/data/file.grib2"
    assert captured[0]["overwrite_existing"] is True
    assert result == get_local_path("my-dataset", "/data/file.grib2")


def test_http_download_to_disk_with_byte_ranges(tmp_path: Path) -> None:
    captured_ranges: list = []

    def fake_download_to_disk(
        store: object,
        path: str,
        local_path: Path,
        *,
        byte_ranges: tuple[Sequence[int], Sequence[int]] | None,
        overwrite_existing: bool,
    ) -> None:
        captured_ranges.append(byte_ranges)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"data")

    byte_ranges = ([0, 100], [50, 200])
    with patch.object(download_module, "download_to_disk", fake_download_to_disk):
        http_download_to_disk(
            "https://example.com/data/file.grib2",
            "my-dataset",
            byte_ranges=byte_ranges,
        )

    assert captured_ranges[0] == byte_ranges
