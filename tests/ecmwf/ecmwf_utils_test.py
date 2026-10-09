from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, call

import obstore
import pytest
from obstore.exceptions import GenericError, PermissionDeniedError
from obstore.store import HTTPStore

from reformatters.ecmwf.ecmwf_utils import (
    EcmwfOpenDataSource,
    ecmwf_download_with_fallback,
)


@pytest.fixture
def missing_object_server() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"<Error><Code>NoSuchKey</Code></Error>")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def _http_error_detail(url: str, status: str = "404 Not Found") -> str:
    return (
        f"Object at location forecast.index not found: Error performing GET {url}"
        f" in 612.648µs - Server returned non-2xx status code: {status}:"
        " <Error><Code>NoSuchKey</Code></Error>\n\nDebug source:\n"
        'NotFound { path: "forecast.index", source: RetryError(...) }'
    )


@pytest.mark.parametrize("source", ["gcs", "s3"])
@pytest.mark.parametrize("suffix", ["index", "grib2"])
def test_404_warning_is_concise(
    source: EcmwfOpenDataSource,
    suffix: str,
    missing_object_server: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = f"20261007000000-360h-enfo-ef.{suffix}"
    url = f"{missing_object_server}/{path}"
    store = HTTPStore(missing_object_server, client_options={"allow_http": True})
    alternate: EcmwfOpenDataSource = "s3" if source == "gcs" else "gcs"

    def download_one(current: EcmwfOpenDataSource) -> Path:
        if current == source:
            if suffix == "index":
                obstore.get(store, path)
            else:
                obstore.get_ranges(store, path, starts=[0], ends=[10])
        return Path("downloaded.grib2")

    download = Mock(side_effect=download_one)

    result = ecmwf_download_with_fallback((source, alternate), download)

    assert result == Path("downloaded.grib2")
    assert download.call_args_list == [call(source), call(alternate)]
    assert [record.getMessage() for record in caplog.records] == [
        (
            f"ECMWF download from {source!r} failed, will fall back:"
            f" FileNotFoundError: 404 Not Found for {url}"
        )
    ]
    assert caplog.records[0].levelname == "WARNING"


@pytest.mark.parametrize(
    "error",
    [
        PermissionDeniedError(
            _http_error_detail("https://example.com", "403 Forbidden")
        ),
        GenericError(
            _http_error_detail("https://example.com", "503 Service Unavailable")
        ),
        GenericError(
            _http_error_detail("https://example.com", "416 Range Not Satisfiable")
        ),
        FileNotFoundError("local file missing\nfull detail"),
        FileNotFoundError(_http_error_detail("https://example.com", "403 Forbidden")),
    ],
)
def test_other_warnings_keep_full_detail(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    download = Mock(side_effect=[error, Path("downloaded.grib2")])

    assert ecmwf_download_with_fallback(("gcs", "s3"), download) == Path(
        "downloaded.grib2"
    )

    assert [record.getMessage() for record in caplog.records] == [
        f"ECMWF download from 'gcs' failed, will fall back: {error}"
    ]


@pytest.mark.parametrize("exception_type", [FileNotFoundError, GenericError])
@pytest.mark.parametrize("url", ["https://example.com/forecast.index", None])
def test_404_substring_shortens_unfamiliar_formats(
    exception_type: type[Exception],
    url: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = exception_type(f"404 Not Found: {url or 'missing object'}\nfull detail")
    download = Mock(side_effect=[error, Path("downloaded.grib2")])

    ecmwf_download_with_fallback(("gcs", "s3"), download)

    detail = f"{exception_type.__name__}: 404 Not Found"
    if url:
        detail += f" for {url}"
    assert [record.getMessage() for record in caplog.records] == [
        f"ECMWF download from 'gcs' failed, will fall back: {detail}"
    ]


def test_exhausted_sources_raise_original_final_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    errors = [
        FileNotFoundError(
            _http_error_detail(f"https://{source}.example.com/forecast.index")
        )
        for source in ("gcs", "s3")
    ]
    original_details = [str(error) for error in errors]
    download = Mock(side_effect=errors)

    with pytest.raises(FileNotFoundError) as caught:
        ecmwf_download_with_fallback(("gcs", "s3"), download)

    assert caught.value is errors[-1]
    assert [str(error) for error in errors] == original_details
    assert download.call_args_list == [call("gcs"), call("s3")]
    assert len(caplog.records) == 2
    assert all(record.levelname == "WARNING" for record in caplog.records)


def test_success_does_not_warn_or_try_another_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    download = Mock(return_value=Path("downloaded.grib2"))

    assert ecmwf_download_with_fallback(("gcs", "s3"), download) == Path(
        "downloaded.grib2"
    )

    download.assert_called_once_with("gcs")
    assert not caplog.records


def test_recovered_gcs_404_can_be_aggregated(caplog: pytest.LogCaptureFixture) -> None:
    error = FileNotFoundError(_http_error_detail("https://example.com/forecast.index"))
    recovered = Mock()
    download = Mock(side_effect=[error, Path("downloaded.grib2")])

    assert ecmwf_download_with_fallback(
        ("gcs", "s3"), download, on_recovered_gcs_404=recovered
    ) == Path("downloaded.grib2")

    recovered.assert_called_once_with(
        "ECMWF download from 'gcs' failed, will fall back: "
        "FileNotFoundError: 404 Not Found for https://example.com/forecast.index"
    )
    assert not caplog.records


@pytest.mark.parametrize(
    "final_error",
    [
        FileNotFoundError(_http_error_detail("https://s3.example.com/forecast.index")),
        PermissionDeniedError("403 Forbidden\nfull diagnostic detail"),
        GenericError("503 Service Unavailable\nfull diagnostic detail"),
        ValueError("unexpected failure"),
    ],
)
def test_unrecovered_404_stays_visible(
    final_error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    recovered = Mock()
    first_error = FileNotFoundError(_http_error_detail("https://gcs.example.com/file"))
    download = Mock(side_effect=[first_error, final_error])

    with pytest.raises(type(final_error)) as caught:
        ecmwf_download_with_fallback(
            ("gcs", "s3"), download, on_recovered_gcs_404=recovered
        )

    assert caught.value is final_error
    recovered.assert_not_called()
    assert caplog.messages[0] == (
        "ECMWF download from 'gcs' failed, will fall back: "
        "FileNotFoundError: 404 Not Found for https://gcs.example.com/file"
    )
    if isinstance(final_error, ValueError):
        assert len(caplog.records) == 1
    elif isinstance(final_error, FileNotFoundError):
        assert len(caplog.records) == 2
        assert (
            "FileNotFoundError: 404 Not Found for https://s3.example.com"
            in caplog.messages[1]
        )
    else:
        assert caplog.messages[1] == (
            f"ECMWF download from 's3' failed, will fall back: {final_error}"
        )


@pytest.mark.parametrize(
    ("sources", "error"),
    [
        (("gcs", "s3"), PermissionDeniedError("403 Forbidden\nfull detail")),
        (("gcs", "s3"), GenericError("503 Service Unavailable\nfull detail")),
        (("gcs", "s3"), FileNotFoundError("local missing file\nfull detail")),
        (
            ("s3", "gcs"),
            FileNotFoundError("404 Not Found for https://example.com/file"),
        ),
    ],
)
def test_other_recoveries_keep_individual_warnings(
    sources: tuple[EcmwfOpenDataSource, EcmwfOpenDataSource],
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recovered = Mock()
    download = Mock(side_effect=[error, Path("downloaded.grib2")])

    ecmwf_download_with_fallback(sources, download, on_recovered_gcs_404=recovered)

    recovered.assert_not_called()
    assert len(caplog.records) == 1
    assert str(error) in caplog.messages[0]
