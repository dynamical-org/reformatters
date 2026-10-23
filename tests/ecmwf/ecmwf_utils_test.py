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
def test_404_fallback_is_quiet(
    source: EcmwfOpenDataSource,
    suffix: str,
    missing_object_server: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = f"20261007000000-360h-enfo-ef.{suffix}"
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
    assert not caplog.records


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
def test_other_fallbacks_are_quiet(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    download = Mock(side_effect=[error, Path("downloaded.grib2")])

    assert ecmwf_download_with_fallback(("gcs", "s3"), download) == Path(
        "downloaded.grib2"
    )

    assert not caplog.records


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
    assert not caplog.records


def test_success_does_not_warn_or_try_another_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    download = Mock(return_value=Path("downloaded.grib2"))

    assert ecmwf_download_with_fallback(("gcs", "s3"), download) == Path(
        "downloaded.grib2"
    )

    download.assert_called_once_with("gcs")
    assert not caplog.records
