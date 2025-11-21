"""A simple tool to copy files from an FTP server to anywhere that `obstore`
can write to.

Single FTP connections are slow. So this code uses multiple concurrent FTP connections to maximise
throughput. Maximising throughput for a single virtual machine should minimise compute costs.

## Overview of architecture

Instead of manually programming two threadpools, the code uses an `asyncio.TaskGroup` containing
a set of `ftp_worker`s and a set of `obstore_worker`s.

The code uses two MPMC queues. The first queue passes `_FtpFile`s to the `ftp_worker`s. The second
queue passes the `_ObstoreFile`s (containing data) to the `obstore_worker`s.

Each `ftp_worker` keeps an `aioftp.Client` alive until there are no more `_FtpFile`s.
"""

import asyncio
from asyncio import Queue
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import aioftp
from obstore.store import ObjectStore

from reformatters.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class _FtpFile:
    src_ftp_path: PurePosixPath
    dst_obstore_path: str
    n_retries: int = field(default=0, init=False)


@dataclass
class _ObstoreFile:
    ftp_file: _FtpFile
    data: bytes
    n_retries: int = field(default=0, init=False)


async def copy_files_from_ftp_to_obstore(
    ftp_host: str,
    src_ftp_paths: Iterable[PurePosixPath],
    dst_obstore_paths: Iterable[str],
    dst_store: ObjectStore,
    n_ftp_workers: int = 8,
    n_obstore_workers: int = 8,
    ftp_port: int = 21,
) -> None:
    """Copy files from an FTP server to obstore."""
    # Copy _FtpFile objects to the ftp_queue:
    ftp_queue: Queue[_FtpFile] = Queue()
    for src, dst in zip(src_ftp_paths, dst_obstore_paths, strict=True):
        ftp_queue.put_nowait(_FtpFile(src, dst))

    # Set maxsize of Queue to apply back-pressure to the FTP workers.
    obstore_queue: Queue[_ObstoreFile] = Queue(maxsize=n_obstore_workers * 2)

    # --- Start the Pipeline ---
    async with asyncio.TaskGroup() as tg:
        for worker_id in range(n_ftp_workers):
            tg.create_task(
                _ftp_worker(
                    worker_id=worker_id,
                    ftp_host=ftp_host,
                    ftp_port=ftp_port,
                    ftp_queue=ftp_queue,
                    obstore_queue=obstore_queue,
                )
            )

        for worker_id in range(n_obstore_workers):
            tg.create_task(
                _obstore_worker(
                    worker_id,
                    dst_store,
                    obstore_queue,
                )
            )

        log.debug("await _ftp_queue.join()")
        await ftp_queue.join()
        log.debug("joined _ftp_queue!")
        obstore_queue.shutdown()


async def _ftp_worker(
    worker_id: int,
    ftp_host: str,
    ftp_port: int,
    ftp_queue: asyncio.Queue[_FtpFile],
    obstore_queue: asyncio.Queue[_ObstoreFile],
    max_retries: int = 5,
) -> None:
    """A worker that keeps a single FTP connection alive and processes queue
    items.

    Reconnects on failure.
    """
    worker_id_str: str = f"ftp_worker {worker_id}:"
    log.info("%s Starting up...", worker_id_str)
    if max_retries < 1:
        raise ValueError(f"max_retries must be > 0, not {max_retries}!")

    # This outer loop exists to retry if the FTP *connection* fails.
    for retry_attempt in range(max_retries):
        try:
            async with aioftp.Client.context(ftp_host, port=ftp_port) as ftp_client:
                log.info(
                    "%s Connection established and logged in to %s port %d.",
                    worker_id_str,
                    ftp_host,
                    ftp_port,
                )
                await _process_ftp_queue(
                    worker_id_str,
                    ftp_client,
                    ftp_queue,
                    obstore_queue,
                    max_retries,
                )
        except (TimeoutError, OSError, aioftp.StatusCodeError) as e:
            is_last_attempt = retry_attempt == max_retries - 1
            log.warning(
                "%s FTP Connection error after %d connection retries: %s",
                worker_id_str,
                retry_attempt,
                e,
            )
            if ftp_queue.empty():
                log.warning(
                    "%s Connection failed but ftp_queue is empty. Finishing.",
                    worker_id_str,
                )
                break
            elif is_last_attempt:
                msg = (
                    f"{worker_id_str} FTP connection failed after {max_retries} attempts to"
                    f" re-connect. FTP worker {worker_id} is giving up. There are files still on"
                    " the `ftp_queue`. Some files may not be copied!"
                )
                log.error(msg)
                raise Exception(msg) from e
            else:
                log.warning("%s Reconnecting in 5s...", worker_id_str)
                await asyncio.sleep(5)
        else:
            break  # All done!


async def _process_ftp_queue(
    worker_id_str: str,
    ftp_client: aioftp.Client,
    ftp_queue: asyncio.Queue[_FtpFile],
    obstore_queue: asyncio.Queue[_ObstoreFile],
    max_retries: int,
) -> None:
    """Process the FTP queue using an active FTP client."""
    while True:  # Loop through items in ftp_queue.
        try:
            ftp_file: _FtpFile = ftp_queue.get_nowait()
        except asyncio.QueueEmpty:
            log.info("%s ftp_queue is empty. Finishing.", worker_id_str)
            return

        log.info("%s Attempting to download ftp_file=%s", worker_id_str, ftp_file)

        try:
            async with ftp_client.download_stream(ftp_file.src_ftp_path) as stream:
                data = await stream.read()
        except (TimeoutError, OSError):
            log.warning(
                "%s Connection lost processing ftp_file=%s. Re-queueing.",
                worker_id_str,
                ftp_file,
            )
            await ftp_queue.put(ftp_file)
            ftp_queue.task_done()
            raise  # Re-raise to trigger reconnection in _ftp_worker
        except aioftp.StatusCodeError as e:
            _log_ftp_exception(ftp_file, e, worker_id_str)
            if ftp_file.n_retries < max_retries:
                ftp_file.n_retries += 1
                log.warning(
                    "%s WARNING: Putting ftp_file back on queue to retry later.",
                    worker_id_str,
                )
                await ftp_queue.put(ftp_file)
            else:
                log.error("%s ERROR: Giving up on ftp_file", worker_id_str)
            ftp_queue.task_done()
        else:
            log.info("%s Finished downloading ftp_file=%s", worker_id_str, ftp_file)
            await obstore_queue.put(_ObstoreFile(ftp_file=ftp_file, data=data))
            ftp_queue.task_done()


async def _obstore_worker(
    worker_id: int,
    store: ObjectStore,
    obstore_queue: asyncio.Queue[_ObstoreFile],
    max_retries: int = 5,
) -> None:
    """Obstores are designed to work concurrently, so we can share one
    `obstore` between tasks."""
    worker_id_str: str = f"obstore_worker {worker_id}:"
    log.debug("%s Obstore worker starting up.", worker_id_str)
    while True:
        try:
            obstore_file: _ObstoreFile = await obstore_queue.get()
        except asyncio.QueueShutDown:
            log.info("%s obstore_queue has shut down. Worker exiting.", worker_id_str)
            break

        dst_path = obstore_file.ftp_file.dst_obstore_path
        try:
            await store.put_async(dst_path, obstore_file.data)
        except Exception as e:  # noqa: BLE001
            # Catching a broad Exception here is intentional, as obstore can raise various
            # exceptions (network, permission, etc.), and we want to retry on any failure
            # to write to the object store.
            log.warning(
                "%s Exception thrown whilst sending file to obstore: %s, %s",
                worker_id_str,
                obstore_file,
                e,
            )
            if obstore_file.n_retries < max_retries:
                obstore_file.n_retries += 1
                log.warning(
                    "%s Putting obstore_file back on queue to retry later.",
                    worker_id_str,
                )
                await obstore_queue.put(obstore_file)
            else:
                log.error(
                    "%s Giving up on obstore_file after %d retries.",
                    worker_id_str,
                    max_retries,
                )
        else:
            log.info(
                "%s Finished writing to %s after %d retries.",
                worker_id_str,
                dst_path,
                obstore_file.n_retries,
            )
        finally:
            obstore_queue.task_done()


def _log_ftp_exception(
    ftp_file: _FtpFile, e: aioftp.StatusCodeError, ftp_worker_id_str: str
) -> None:
    error_str = f"{ftp_worker_id_str} "
    # Check if the file is missing from the FTP server:
    if (
        isinstance(e, aioftp.StatusCodeError)
        and isinstance(e.received_codes, tuple)
        and len(e.received_codes) == 1
        and e.received_codes[0].matches("550")
    ):
        error_str += f"File not available on FTP server: {ftp_file.src_ftp_path} "

    log.warning(
        "%s ftp_client.download_stream raised exception whilst processing ftp_file=%s %s",
        error_str,
        ftp_file,
        e,
    )
