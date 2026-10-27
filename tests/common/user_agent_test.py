from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from threading import Thread

import icechunk
import numpy as np
import pytest
import zarr

from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
    StoreFactory,
    manifest_append_dim_split,
)
from scripts.validation import utils as validation_utils

CHUNK = np.arange(4, dtype="float32")


class ChunkServer:
    def __init__(self, url_prefix: str) -> None:
        self.url_prefix = url_prefix
        self.user_agents: list[str | None] = []


@pytest.fixture
def chunk_server() -> Iterator[ChunkServer]:
    payload = CHUNK.tobytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            server.user_agents.append(self.headers.get("User-Agent"))
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
        server = ChunkServer(f"http://127.0.0.1:{http_server.server_port}/chunks/")
        thread = Thread(target=http_server.serve_forever)
        thread.start()
        try:
            yield server
        finally:
            http_server.shutdown()
            thread.join()


def _virtual_factory(tmp_path: Path, url_prefix: str) -> StoreFactory:
    return StoreFactory(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        dataset_id="test-dataset",
        template_config_version="v1.0",
        icechunk_virtual_config=IcechunkVirtualConfig(
            containers=(
                icechunk.VirtualChunkContainer(url_prefix, icechunk.http_store()),
            ),
            manifest_split=manifest_append_dim_split(split_size=1, dim="init_time"),
        ),
    )


def _write_virtual_array(factory: StoreFactory, url_prefix: str) -> None:
    store = factory.primary_store(writable=True)
    assert isinstance(store, icechunk.IcechunkStore)
    zarr.create_array(
        store,
        name="a",
        shape=CHUNK.shape,
        chunks=CHUNK.shape,
        dtype=CHUNK.dtype,
        compressors=None,
        fill_value=np.nan,
    )
    store.set_virtual_ref("a/c/0", f"{url_prefix}chunk", offset=0, length=CHUNK.nbytes)
    store.session.commit("virtual ref")


def _expected_user_agent(workload: str) -> str:
    return (
        f"dynamical-reformatters/{version('reformatters')} "
        f"icechunk/{icechunk.__version__} (workload={workload}; array=a; chunk=0)"
    )


def test_store_factory_virtual_chunk_fetch_user_agent(
    tmp_path: Path, chunk_server: ChunkServer
) -> None:
    factory = _virtual_factory(tmp_path, chunk_server.url_prefix)
    _write_virtual_array(factory, chunk_server.url_prefix)

    array = zarr.open_array(factory.primary_store(), path="a")

    np.testing.assert_array_equal(array[:], CHUNK)
    assert chunk_server.user_agents == [_expected_user_agent("test-dataset")]


def test_validation_report_virtual_chunk_fetch_user_agent(
    tmp_path: Path, chunk_server: ChunkServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _virtual_factory(tmp_path, chunk_server.url_prefix)
    _write_virtual_array(factory, chunk_server.url_prefix)
    factory.persist_virtual_config()
    [(_role, _path, storage)] = factory._icechunk_storages()
    monkeypatch.setattr(validation_utils, "_icechunk_storage", lambda url: storage)

    store = validation_utils.open_icechunk_readonly("unused.icechunk")
    array = zarr.open_array(store, path="a")

    np.testing.assert_array_equal(array[:], CHUNK)
    assert chunk_server.user_agents == [_expected_user_agent("validation-report")]
