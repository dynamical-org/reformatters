from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import obstore.store
import pytest

from reformatters.common import download as download_module
from reformatters.common.download import (
    DOWNLOAD_DIR,
    _httpx_get_with_retry,
    _parse_multipart_byteranges,
    download_to_disk,
    get_local_path,
    http_download_to_disk,
    http_store,
    httpx_download_to_disk,
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


# --- _parse_multipart_byteranges tests ---


def test_parse_multipart_byteranges_two_parts() -> None:
    boundary = "ABCD1234"
    body = (
        b"--ABCD1234\r\n"
        b"Content-Range: bytes 0-4/100\r\n"
        b"\r\n"
        b"hello"
        b"\r\n"
        b"--ABCD1234\r\n"
        b"Content-Range: bytes 10-14/100\r\n"
        b"\r\n"
        b"world"
        b"\r\n"
        b"--ABCD1234--\r\n"
    )
    content_type = f"multipart/byteranges; boundary={boundary}"
    result = _parse_multipart_byteranges(body, content_type)
    assert result == b"helloworld"


def test_parse_multipart_byteranges_single_part() -> None:
    body = (
        b"--boundary99\r\nContent-Range: bytes 0-2/10\r\n\r\nabc\r\n--boundary99--\r\n"
    )
    result = _parse_multipart_byteranges(
        body, "multipart/byteranges; boundary=boundary99"
    )
    assert result == b"abc"


def test_parse_multipart_byteranges_binary_data() -> None:
    binary_data = bytes(range(256))
    body = (
        b"--B\r\n"
        b"Content-Range: bytes 0-255/256\r\n"
        b"\r\n" + binary_data + b"\r\n"
        b"--B--\r\n"
    )
    result = _parse_multipart_byteranges(body, "multipart/byteranges; boundary=B")
    assert result == binary_data


def test_parse_multipart_byteranges_no_boundary_raises() -> None:
    with pytest.raises(AssertionError, match="No boundary"):
        _parse_multipart_byteranges(b"data", "application/octet-stream")


# --- httpx_download_to_disk tests ---


def _make_httpx_response(
    status_code: int = 200,
    content: bytes = b"",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    response = httpx.Response(
        status_code=status_code,
        content=content,
        headers=headers or {},
        request=httpx.Request("GET", "https://example.com/test"),
    )
    return response


def test_httpx_download_to_disk_no_byte_ranges(tmp_path: Path) -> None:
    file_content = b"full file content here"

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_bytes = Mock(return_value=iter([file_content]))
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    mock_client = Mock()
    mock_client.stream = Mock(return_value=mock_response)

    with (
        patch.object(download_module, "_httpx_client", return_value=mock_client),
        patch.object(download_module, "DOWNLOAD_DIR", tmp_path),
    ):
        result = httpx_download_to_disk(
            "https://example.com/data/file.grib2", "my-dataset"
        )

    assert result.exists()
    assert result.read_bytes() == file_content


def test_httpx_download_to_disk_with_byte_ranges_multipart(tmp_path: Path) -> None:
    multipart_body = (
        b"--BOUNDARY\r\n"
        b"Content-Range: bytes 0-4/100\r\n"
        b"\r\n"
        b"AAAAA"
        b"\r\n"
        b"--BOUNDARY\r\n"
        b"Content-Range: bytes 50-54/100\r\n"
        b"\r\n"
        b"BBBBB"
        b"\r\n"
        b"--BOUNDARY--\r\n"
    )
    response = _make_httpx_response(
        status_code=206,
        content=multipart_body,
        headers={"content-type": "multipart/byteranges; boundary=BOUNDARY"},
    )

    with (
        patch.object(download_module, "_httpx_get_with_retry", return_value=response),
        patch.object(download_module, "DOWNLOAD_DIR", tmp_path),
    ):
        result = httpx_download_to_disk(
            "https://example.com/data/file.grib2",
            "my-dataset",
            byte_ranges=([0, 50], [5, 55]),
        )

    assert result.exists()
    assert result.read_bytes() == b"AAAAABBBBB"


def test_httpx_download_to_disk_with_byte_ranges_single_range(tmp_path: Path) -> None:
    response = _make_httpx_response(
        status_code=206,
        content=b"single range data",
        headers={"content-type": "application/octet-stream"},
    )

    with (
        patch.object(download_module, "_httpx_get_with_retry", return_value=response),
        patch.object(download_module, "DOWNLOAD_DIR", tmp_path),
    ):
        result = httpx_download_to_disk(
            "https://example.com/data/file.grib2",
            "my-dataset",
            byte_ranges=([100], [200]),
        )

    assert result.exists()
    assert result.read_bytes() == b"single range data"


def test_httpx_download_to_disk_builds_correct_range_header(tmp_path: Path) -> None:
    """Verify the Range header uses inclusive end (end - 1 from exclusive grib index ends)."""
    captured_headers: list[dict[str, str] | None] = []

    def fake_get_with_retry(
        url: str, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        captured_headers.append(headers)
        return _make_httpx_response(
            status_code=206,
            content=b"data",
            headers={"content-type": "application/octet-stream"},
        )

    with (
        patch.object(download_module, "_httpx_get_with_retry", fake_get_with_retry),
        patch.object(download_module, "DOWNLOAD_DIR", tmp_path),
    ):
        httpx_download_to_disk(
            "https://example.com/data/file.grib2",
            "my-dataset",
            byte_ranges=([0, 100, 500], [50, 200, 600]),
        )

    assert captured_headers[0] is not None
    # Ends are exclusive from grib index, so HTTP Range should be end-1 (inclusive)
    assert captured_headers[0]["Range"] == "bytes=0-49, 100-199, 500-599"


def test_httpx_download_to_disk_cleans_up_on_error(tmp_path: Path) -> None:
    with (
        patch.object(
            download_module,
            "_httpx_get_with_retry",
            side_effect=httpx.ConnectError("failed"),
        ),
        patch.object(download_module, "DOWNLOAD_DIR", tmp_path),
        pytest.raises(httpx.ConnectError),
    ):
        httpx_download_to_disk(
            "https://example.com/data/file.grib2",
            "my-dataset",
            byte_ranges=([0], [100]),
        )

    # No temp or final files should remain
    all_files = list(tmp_path.rglob("*"))
    data_files = [f for f in all_files if f.is_file()]
    assert len(data_files) == 0


def test_httpx_download_to_disk_with_suffix(tmp_path: Path) -> None:
    response = _make_httpx_response(
        status_code=206,
        content=b"data",
        headers={"content-type": "application/octet-stream"},
    )

    with (
        patch.object(download_module, "_httpx_get_with_retry", return_value=response),
        patch.object(download_module, "DOWNLOAD_DIR", tmp_path),
    ):
        result = httpx_download_to_disk(
            "https://example.com/data/file.grib2",
            "my-dataset",
            byte_ranges=([0], [100]),
            local_path_suffix="-abc123",
        )

    assert result.name == "file.grib2-abc123"


# --- _httpx_get_with_retry tests ---


def test_httpx_get_with_retry_retries_on_5xx() -> None:
    call_count = 0

    def mock_get(url: str, **kwargs: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return _make_httpx_response(status_code=503, content=b"unavailable")
        return _make_httpx_response(status_code=200, content=b"ok")

    mock_client = Mock()
    mock_client.get = mock_get

    with (
        patch.object(download_module, "_httpx_client", return_value=mock_client),
        patch.object(download_module.time, "sleep"),
    ):
        response = _httpx_get_with_retry("https://example.com/test")

    assert call_count == 3
    assert response.status_code == 200


def test_httpx_get_with_retry_retries_on_transport_error() -> None:
    call_count = 0

    def mock_get(url: str, **kwargs: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.ConnectError("connection refused")
        return _make_httpx_response(status_code=200, content=b"ok")

    mock_client = Mock()
    mock_client.get = mock_get

    with (
        patch.object(download_module, "_httpx_client", return_value=mock_client),
        patch.object(download_module.time, "sleep"),
    ):
        response = _httpx_get_with_retry("https://example.com/test")

    assert call_count == 2
    assert response.status_code == 200


def test_httpx_get_with_retry_raises_on_4xx() -> None:
    mock_client = Mock()
    mock_client.get = Mock(
        return_value=_make_httpx_response(status_code=404, content=b"not found")
    )

    with (
        patch.object(download_module, "_httpx_client", return_value=mock_client),
        pytest.raises(httpx.HTTPStatusError),
    ):
        _httpx_get_with_retry("https://example.com/test")
