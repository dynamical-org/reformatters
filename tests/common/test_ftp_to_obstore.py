import asyncio
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

import aioftp
import obstore

from reformatters.common.ftp_to_obstore import copy_files_from_ftp_to_obstore


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
        (root / "test_file.txt").write_text("Hello, World!")

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


async def _run_ftp_test() -> None:
    async with ftp_server_context() as (host, port, _):
        src_paths = [PurePosixPath("test_file.txt")]
        dst_paths = ["test_file.txt"]
        dst_store = obstore.store.MemoryStore()

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
        response = await dst_store.get_async("test_file.txt")
        data = response.bytes()
        assert data == b"Hello, World!"


def test_ftp_to_obstore() -> None:
    asyncio.run(_run_ftp_test())
