import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

import aioftp
import obstore
import pytest

from reformatters.common.ftp_to_obstore import copy_files_from_ftp_to_obstore

if TYPE_CHECKING:
    from obstore import PutResult

TEST_FILE_NAME: Final = "test_file.txt"
TEST_FILE_CONTENT: Final = "Hello, World!"
SLEEP_BETWEEN_RETRIES: Final = 0.001  # seconds


@asynccontextmanager
async def ftp_server_context() -> AsyncIterator[tuple[str, int, Path]]:
    """Starts an aioftp server and yields (host, port, root_dir).

    The security risks are negligible. The combination of localhost
    binding, ephemeral ports, temporary storage, short lifetime, and
    read-only permissions makes this a very safe test pattern.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        # Create a dummy file
        (root / TEST_FILE_NAME).write_text(TEST_FILE_CONTENT)

        # Configure users
        read_only_permissions = aioftp.Permission(
            path=PurePosixPath(root), readable=True, writable=False
        )
        users = [aioftp.User(base_path=root, permissions=[read_only_permissions])]

        # Start server on random port
        server = aioftp.Server(users=users)
        # Bind to localhost on a random port
        await server.start("127.0.0.1", 0)

        host, port = server.server.sockets[0].getsockname()

        try:
            yield host, port, root
        finally:
            await server.close()


@pytest.fixture
def src_paths() -> list[PurePosixPath]:
    return [PurePosixPath(TEST_FILE_NAME)]


@pytest.fixture
def dst_paths() -> list[str]:
    return [TEST_FILE_NAME]


@pytest.fixture
def dst_store() -> obstore.store.MemoryStore:
    return obstore.store.MemoryStore()


async def test_ftp_to_obstore(
    src_paths: list[PurePosixPath],
    dst_paths: list[str],
    dst_store: obstore.store.MemoryStore,
) -> None:
    async with ftp_server_context() as (host, port, _):
        await copy_files_from_ftp_to_obstore(
            ftp_host=host,
            src_ftp_paths=src_paths,
            dst_obstore_paths=dst_paths,
            dst_store=dst_store,
            ftp_port=port,
            n_ftp_workers=2,
            n_obstore_workers=2,
        )

        # Verify the file was copied
        response = await dst_store.get_async(TEST_FILE_NAME)
        data = response.bytes()
        assert data == TEST_FILE_CONTENT.encode("utf-8")


async def test_ftp_connection_failure(
    src_paths: list[PurePosixPath],
    dst_paths: list[str],
    dst_store: obstore.store.MemoryStore,
) -> None:
    """Check that an Exception is thrown if the FTP server doesn't exist."""
    with pytest.raises(ExceptionGroup) as exc_info:
        await copy_files_from_ftp_to_obstore(
            ftp_host="127.0.0.1",
            src_ftp_paths=src_paths,
            dst_obstore_paths=dst_paths,
            dst_store=dst_store,
            ftp_port=21212,
            n_ftp_workers=1,
            n_obstore_workers=1,
            sleep_between_retries=SLEEP_BETWEEN_RETRIES,
        )

    assert any(
        "FTP connection failed after" in str(e) for e in exc_info.value.exceptions
    )


async def test_obstore_write_failure(
    src_paths: list[PurePosixPath],
    dst_paths: list[str],
) -> None:
    class FailingMemoryStore(obstore.store.MemoryStore):
        async def put_async(  # type: ignore[override]
            self,
            *_args: tuple[Any, ...],
            **_kwargs: dict[str, Any],
        ) -> "PutResult":
            raise ValueError("Simulated obstore write failure")

    dst_store = FailingMemoryStore()

    async with ftp_server_context() as (host, port, _):
        await copy_files_from_ftp_to_obstore(
            ftp_host=host,
            src_ftp_paths=src_paths,
            dst_obstore_paths=dst_paths,
            dst_store=dst_store,
            ftp_port=port,
            n_ftp_workers=1,
            n_obstore_workers=1,
            sleep_between_retries=SLEEP_BETWEEN_RETRIES,
        )

    with pytest.raises(FileNotFoundError):
        await dst_store.get_async(TEST_FILE_NAME)


async def test_ftp_file_not_found(
    caplog: pytest.LogCaptureFixture,
    dst_store: obstore.store.MemoryStore,
) -> None:
    """Test for attempting to load a file from FTP that doesn't exist."""
    src_paths = [PurePosixPath("non_existent_file.txt")]
    dst_paths = ["non_existent_file.txt"]

    async with ftp_server_context() as (host, port, _):
        await copy_files_from_ftp_to_obstore(
            ftp_host=host,
            src_ftp_paths=src_paths,
            dst_obstore_paths=dst_paths,
            dst_store=dst_store,
            ftp_port=port,
            n_ftp_workers=1,
            n_obstore_workers=1,
            sleep_between_retries=SLEEP_BETWEEN_RETRIES,
        )

    with pytest.raises(FileNotFoundError):
        await dst_store.get_async("non_existent_file.txt")

    assert any(
        "File not available on FTP server: non_existent_file.txt" in record.message
        for record in caplog.records
        if record.levelname in {"ERROR", "WARNING"}
    )
