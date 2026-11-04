import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import icechunk
import numpy as np
import obstore
import pytest
import zarr

from reformatters.common import download, kubernetes, storage
from reformatters.common.config import Config, Env
from scripts.validation import utils

VALUES = np.arange(16, dtype="<f4")
SOURCE_PREFIX = "s3://source/"
SOURCE_KEYS = {
    "access_key_id": "synthetic-reader",
    "secret_access_key": "synthetic-reader-secret",
    "session_token": "synthetic-reader-token",
}
WRITER_KEYS = {
    "access_key_id": "synthetic-writer",
    "secret_access_key": "synthetic-writer-secret",
    "session_token": "synthetic-writer-token",
}


@pytest.fixture
def source_server() -> Iterator[tuple[str, list[str]]]:
    # This fixture checks signed identity and ranges, not the SigV4 cryptography.
    requests: list[str] = []
    objects = {"values": VALUES.tobytes(), "index": b'{"offset": 0, "length": 64}'}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            authorization = self.headers.get("Authorization", "")
            if (
                not authorization.startswith("AWS4-HMAC-SHA256 ")
                or "Credential=synthetic-reader/" not in authorization
                or self.headers.get("x-amz-security-token")
                != SOURCE_KEYS["session_token"]
            ):
                self.send_error(403)
                return
            requests.append(self.path)
            parsed = urlparse(self.path)
            if "list-type=" in parsed.query:
                payload = (
                    '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                    "<Name>source</Name><IsTruncated>false</IsTruncated>"
                    + "".join(
                        f"<Contents><Key>{key}</Key><Size>{len(value)}</Size>"
                        "<LastModified>2026-01-01T00:00:00Z</LastModified>"
                        '<ETag>"fixture"</ETag></Contents>'
                        for key, value in objects.items()
                    )
                    + "</ListBucketResult>"
                ).encode()
            else:
                key = parsed.path.removeprefix("/source/")
                if key not in objects:
                    self.send_error(404)
                    return
                payload = objects[key]
            size = len(payload)
            byte_range = self.headers.get("Range")
            self.send_response(206 if byte_range else 200)
            if byte_range:
                start, end = byte_range.removeprefix("bytes=").split("-")
                first, last = int(start), int(end) if end else size - 1
                payload = payload[first : last + 1]
                self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("ETag", '"fixture"')
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://localhost:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        download.s3_store_from_secret.cache_clear()
        download.s3_store.cache_clear()


def test_signed_source_listing_index_and_range(
    source_server: tuple[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint, requests = source_server
    monkeypatch.setattr(kubernetes, "load_secret", lambda _: SOURCE_KEYS)
    store = download.s3_store_from_secret(
        SOURCE_PREFIX,
        region="auto",
        k8s_secret_name="reader",  # noqa: S106
        endpoint=endpoint,
    )
    with download.source_access():
        entries = [entry for page in obstore.list(store) for entry in page]
        download.download_to_disk(store, "index", tmp_path / "index")
        data = bytes(obstore.get_range(store, "values", start=4, end=12))
    assert {entry["path"] for entry in entries} == {"values", "index"}
    assert json.loads((tmp_path / "index").read_bytes())["length"] == VALUES.nbytes
    assert data == VALUES.tobytes()[4:12]
    assert all(path.startswith("/source") for path in requests)
    with pytest.raises(FileNotFoundError), download.source_access():
        obstore.get(store, "missing")
    wrong = download.s3_store(
        SOURCE_PREFIX,
        region="auto",
        endpoint=endpoint,
        skip_signature=False,
        **WRITER_KEYS,
    )
    with pytest.raises(download.SourceAccessError), download.source_access():
        obstore.get_range(wrong, "values", start=0, end=4)


def test_factory_and_validation_decode_with_separate_credentials(
    source_server: tuple[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint, requests = source_server
    writer_options = {**WRITER_KEYS, "region": "auto", "endpoint_url": endpoint}
    secrets = {"reader": SOURCE_KEYS, "writer": writer_options}
    monkeypatch.setattr(kubernetes, "load_secret", secrets.__getitem__)
    monkeypatch.setattr(Config, "env", Env.prod)
    output = icechunk.local_filesystem_storage(str(tmp_path / "repository"))
    output_calls: list[dict[str, object]] = []

    def output_storage(**kwargs: object) -> icechunk.Storage:
        output_calls.append(kwargs)
        return output

    monkeypatch.setattr(icechunk, "s3_storage", output_storage)
    container = icechunk.VirtualChunkContainer(
        SOURCE_PREFIX,
        icechunk.s3_store(
            region="auto",
            endpoint_url=endpoint,
            allow_http=True,
            s3_compatible=True,
            force_path_style=True,
            anonymous=True,
        ),
    )
    virtual_config = storage.IcechunkVirtualConfig(
        containers=(container,),
        manifest_split=storage.manifest_append_dim_split(split_size=1, dim="time"),
        source_secret_names={SOURCE_PREFIX: "reader"},
    )
    factory = storage.StoreFactory(
        dataset_id="synthetic",
        template_config_version="1",
        primary_storage_config=storage.StorageConfig(
            base_path="s3://output",
            format=storage.DatasetFormat.ICECHUNK,
            k8s_secret_name="writer",  # noqa: S106
        ),
        icechunk_virtual_config=virtual_config,
    )
    store = factory.primary_store(writable=True)
    assert isinstance(store, icechunk.IcechunkStore)
    zarr.create_array(
        store=store,
        shape=(16,),
        dtype="float32",
        chunks=(16,),
        compressors=None,
        dimension_names=("time",),
    )
    store.set_virtual_ref(
        "c/0", f"{SOURCE_PREFIX}values", offset=0, length=VALUES.nbytes
    )
    store.session.commit("synthetic virtual data")
    factory.persist_virtual_config()
    np.testing.assert_array_equal(zarr.open_array(factory.primary_store())[:], VALUES)
    assert output_calls
    assert all(
        call["access_key_id"] == WRITER_KEYS["access_key_id"]
        and call["session_token"] == WRITER_KEYS["session_token"]
        for call in output_calls
    )
    monkeypatch.setenv(
        "REFORMATTERS_ICECHUNK_ACCESS",
        json.dumps(
            {
                factory.primary_url(): {
                    "output_secret_name": "writer",
                    "source_secret_names": {SOURCE_PREFIX: "reader"},
                }
            }
        ),
    )
    assert utils.is_virtual_store(factory.primary_url())
    for _ in range(2):
        np.testing.assert_array_equal(
            zarr.open_array(utils.open_icechunk_readonly(factory.primary_url()))[:],
            VALUES,
        )
    assert requests
    pod = kubernetes.Job(
        command=["update"],
        image="synthetic",
        dataset_id="synthetic",
        cpu="1",
        memory="1G",
        workers_total=1,
        parallelism=1,
        secret_names=factory.k8s_secret_names(),
    ).as_kubernetes_object()["spec"]["template"]["spec"]
    assert {
        volume["secret"]["secretName"]
        for volume in pod["volumes"]
        if "secret" in volume
    } == {
        "writer",
        "reader",
    }
    for path in (tmp_path / "repository").rglob("*"):
        if path.is_file():
            assert all(
                value.encode() not in path.read_bytes()
                for value in [*SOURCE_KEYS.values(), *WRITER_KEYS.values()]
            )
    for credentials in ({}, {SOURCE_PREFIX: "writer"}):
        authorization = storage.virtual_chunk_credentials([container], credentials)
        repo = icechunk.Repository.open(
            output, authorize_virtual_chunk_access=authorization
        )
        with pytest.raises(icechunk.IcechunkError):
            zarr.open_array(repo.readonly_session("main").store)[:]
