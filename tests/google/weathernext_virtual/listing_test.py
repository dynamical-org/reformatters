import json
from base64 import b64encode
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import httpx
import icechunk
import numpy as np
import pytest
import zarr
from kubernetes.client.exceptions import ApiException

from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
    StoreFactory,
    manifest_append_dim_split,
)
from reformatters.google.weathernext_virtual import listing
from reformatters.google.weathernext_virtual.listing import (
    OBJECTS_LOCATION,
    PROXY_LOCATION_PREFIX,
    NativeObjectMetadata,
    ObjectListingQuery,
    list_objects,
    weathernext_read_chunk_containers,
    weathernext_virtual_chunk_containers,
)


@pytest.mark.parametrize("glob", [None, "store/temperature_*/c/0/0/0"])
@pytest.mark.parametrize("delimiter", [None, "/"])
def test_listing_delimiter_is_independent_and_placeholders_are_skipped(
    glob: str | None, delimiter: str | None
) -> None:
    prefix = "store/temperature_"
    key = prefix + "mean/c/0/0/0"
    md5 = bytes(range(16))
    client = Mock()
    request = httpx.Request("GET", OBJECTS_LOCATION)
    client.get.side_effect = [
        httpx.Response(
            200,
            request=request,
            json={
                "items": [{"name": prefix + "mean/", "size": "0"}],
                "nextPageToken": "second",
            },
        ),
        httpx.Response(
            200,
            request=request,
            json={
                "items": [
                    {"name": key, "size": "123", "md5Hash": b64encode(md5).decode()}
                ]
            },
        ),
    ]

    assert list_objects(client, ObjectListingQuery(prefix, glob, delimiter)) == {
        PROXY_LOCATION_PREFIX + key: NativeObjectMetadata(
            123, '"000102030405060708090a0b0c0d0e0f"'
        )
    }
    params = {"prefix": prefix, "maxResults": "1000"}
    if glob is not None:
        params["matchGlob"] = glob
    if delimiter is not None:
        params["delimiter"] = delimiter
    assert [(call.args, call.kwargs) for call in client.get.call_args_list] == [
        ((OBJECTS_LOCATION,), {"params": params}),
        ((OBJECTS_LOCATION,), {"params": params | {"pageToken": "second"}}),
    ]


def test_zero_byte_non_placeholder_is_rejected() -> None:
    client = Mock()
    client.get.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", OBJECTS_LOCATION),
        json={"items": [{"name": "store/temperature/0.0", "size": "0"}]},
    )
    with pytest.raises(AssertionError, match="invalid object size"):
        list_objects(client, ObjectListingQuery("store/temperature/"))


TOKEN = "test-service-token"  # noqa: S105
CHUNK = np.arange(4, dtype="float32")


@pytest.fixture
def chunk_server_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[str, list[str | None]]]:
    payload = CHUNK.tobytes()
    tokens: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            tokens.append(self.headers.get("X-Dynamical-Service-Token"))
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes 0-{len(payload) - 1}/{len(payload)}"
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http_server:
        prefix = f"http://127.0.0.1:{http_server.server_port}/chunks/"
        monkeypatch.setattr(listing, "PROXY_LOCATION_PREFIX", prefix)
        thread = Thread(target=http_server.serve_forever)
        thread.start()
        try:
            yield prefix, tokens
        finally:
            http_server.shutdown()
            thread.join()


def mock_service_token_secret(
    monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    def load_secret(name: str) -> dict[str, str]:
        assert name == "wn-service-token"
        return {} if token is None else {"WN_SERVICE_TOKEN": token}

    monkeypatch.setattr(listing.kubernetes, "load_secret", load_secret)


@pytest.mark.parametrize("token", [TOKEN, "", None])
def test_readonly_chunk_reads_send_service_token_without_persisting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunk_server_prefix: tuple[str, list[str | None]],
    token: str | None,
) -> None:
    prefix, received_tokens = chunk_server_prefix
    mock_service_token_secret(monkeypatch, token)
    factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        dataset_id="test-dataset",
        template_config_version="v1.0",
        icechunk_virtual_config=IcechunkVirtualConfig(
            containers=weathernext_virtual_chunk_containers(),
            read_containers=weathernext_read_chunk_containers,
            manifest_split=manifest_append_dim_split(split_size=1, dim="init_time"),
        ),
    )
    writable = factory.primary_store(writable=True)
    assert isinstance(writable, icechunk.IcechunkStore)
    zarr.create_array(
        writable,
        name="a",
        shape=CHUNK.shape,
        chunks=CHUNK.shape,
        dtype=CHUNK.dtype,
        compressors=None,
        fill_value=np.nan,
    )
    writable.set_virtual_ref("a/c/0", f"{prefix}chunk", offset=0, length=CHUNK.nbytes)
    writable.session.commit("virtual ref")
    factory.persist_virtual_config()

    array = zarr.open_array(factory.primary_store(), path="a")

    np.testing.assert_array_equal(array[:], CHUNK)
    assert received_tokens == [token or None]
    [(_role, _path, storage)] = factory._icechunk_storages()
    persisted = icechunk.Repository.fetch_config(storage)
    assert persisted is not None
    assert "X-Dynamical-Service-Token" not in repr(persisted)
    assert not any(
        TOKEN.encode() in path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )


@pytest.mark.parametrize("status", [403, 404])
def test_unreadable_service_token_secret_falls_back_to_public(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    def load_secret(name: str) -> dict[str, str]:
        raise ApiException(status=status)

    monkeypatch.setattr(listing.kubernetes, "load_secret", load_secret)
    assert listing.service_token_headers() == {}


@pytest.mark.parametrize(
    "error", [ApiException(status=500), json.JSONDecodeError("bad", "", 0)]
)
def test_other_service_token_secret_failures_propagate(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def load_secret(name: str) -> dict[str, str]:
        raise error

    monkeypatch.setattr(listing.kubernetes, "load_secret", load_secret)
    with pytest.raises(type(error)):
        listing.service_token_headers()
