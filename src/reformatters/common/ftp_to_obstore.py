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
from dataclasses import field
from pathlib import PurePosixPath

import aioftp
from obstore.store import ObjectStore
from pydantic.dataclasses import dataclass

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
    max_retries: int = 5,
) -> None:
    """Copy files from an FTP server to obstore.

    Args:
        ftp_host: The FTP host URL, e.g. "opendata.dwd.de"
        src_ftp_paths: A list of the FTP files to download from the FTP server.
            Must be the same length as `dst_obstore_paths`.
        dst_obstore_paths: A list of obstore locations to save.
            Must be the same length as `src_ftp_paths`.
        dst_store: The destination `ObjectStore`.
        n_ftp_workers: The number of concurrent FTP workers.
        n_obstore_workers: The number of concurrent `obstore` workers.
        ftp_port: The port for the FTP connection.
        max_retries: The maximum number of times to retry before giving up.
            See the "Error handling" section below for more details.


    Error handling:
        If the FTP connection is lost then the code tries `max_retries` times to reconnect.
        If the connection fails `max_retries` times then the code raises an exception
        (see the Raises section below). A warning is logged on every connection failure.

        If an individual file fails to be downloaded from the FTP server, then the code tries `max_retries`
        times per file, and logs a warning on each failure. If the file still cannot be downloaded
        after `max_retries` times then the code logs an error but does not raise an exception (so as not to
        interrupt downloading the remaining files in `src_ftp_paths`). The code takes the same
        approach for _saving_ files to the `dst_obstore_path`: The code tries `max_retries` per
        file, and logs a warning on each failure, but does not raise an Exception.

    Raises:
        A `TimeoutError`, `OSError`, or `StatusCodeError` if the FTP connection fails after
        `max_retries` attempts.
    """
    # Put _FtpFile objects on the ftp_queue:
    ftp_queue: Queue[_FtpFile] = Queue()
    for src, dst in zip(src_ftp_paths, dst_obstore_paths, strict=True):
        ftp_queue.put_nowait(_FtpFile(src, dst))

    # Set maxsize of Queue to apply back-pressure to the FTP workers.
    obstore_queue: Queue[_ObstoreFile] = Queue(maxsize=n_obstore_workers * 2)

    # --- Start the Pipeline ---
    async with asyncio.TaskGroup() as task_group:
        for worker_id in range(n_ftp_workers):
            task_group.create_task(
                _ftp_worker(
                    worker_id=worker_id,
                    ftp_host=ftp_host,
                    ftp_port=ftp_port,
                    ftp_queue=ftp_queue,
                    obstore_queue=obstore_queue,
                    max_retries=max_retries,
                )
            )

        for worker_id in range(n_obstore_workers):
            task_group.create_task(
                _obstore_worker(
                    worker_id=worker_id,
                    store=dst_store,
                    obstore_queue=obstore_queue,
                    max_retries=max_retries,
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

    Tries `max_retries` times to reconnect if the FTP connection is lost. If the connection
    fails `max_retries` times then raises an exception (see the Raises section below).

    Args:
        worker_id: The unique ID of this FTP worker.
        ftp_host: The FTP host URL, e.g. "opendata.dwd.de"
        ftp_port: The port for FTP connections.
        ftp_queue: The MPMC queue of files to download. Shared across FTP workers.
            This is the input to the FTP workers.
        obstore_queue: The MPMC queue of bytes that have been downloaded, and their destination
            paths on object storage. This is the output of the FTP workers.
        max_retries: The maximum number of times to try downloading each FTP file before giving up.
            max_retries must be >= 1.

    Raises:
        A `TimeoutError`, `OSError`, or `StatusCodeError` if the FTP connection fails after
        `max_retries` attempts.
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
                    "%s Connection failed but ftp_queue is empty. FTP worker finished.",
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
    """Process the FTP queue using an active FTP client.

    Args:
        worker_id_str: The unique ID of this FTP worker.
        ftp_client: The open FTP connection.
        ftp_queue: The MPMC queue of files to download. Shared across FTP workers.
            This is the input to the FTP workers.
        obstore_queue: The MPMC queue of bytes that have been downloaded, and their destination
            paths on object storage. This is the output of the FTP workers.
        max_retries: The maximum number of times to try downloading each FTP file before giving up.
            max_retries must be >= 1.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be > 0, not {max_retries}!")

    while True:  # Loop through items in ftp_queue.
        try:
            ftp_file: _FtpFile = ftp_queue.get_nowait()
        except asyncio.QueueEmpty:
            log.info("%s ftp_queue is empty. FTP worker finished.", worker_id_str)
            return

        log.debug("%s Attempting to download %s", worker_id_str, ftp_file.src_ftp_path)

        try:
            async with ftp_client.download_stream(ftp_file.src_ftp_path) as stream:
                data = await stream.read()
        except (TimeoutError, OSError):
            log.warning(
                "%s FTP connection lost whilst processing ftp_file=%s. Re-queueing.",
                worker_id_str,
                ftp_file,
            )
            await ftp_queue.put(ftp_file)
            raise  # Re-raise to trigger reconnection in _ftp_worker.
            # Note that the `finally` block is called before the exception propagates upwards.
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
                log.exception(
                    "%s ERROR: Giving up downloading ftp_file %s. Exception: %s",
                    worker_id_str,
                    ftp_file,
                    e,
                )
        else:
            log.info("%s Finished downloading %s", worker_id_str, ftp_file.src_ftp_path)
            await obstore_queue.put(_ObstoreFile(ftp_file=ftp_file, data=data))
        finally:
            ftp_queue.task_done()


async def _obstore_worker(
    worker_id: int,
    store: ObjectStore,
    obstore_queue: asyncio.Queue[_ObstoreFile],
    max_retries: int = 5,
) -> None:
    """Obstores are designed to work concurrently, so we can share one
    `obstore` between tasks.

    Args:
        worker_id: The unique ID of this obstore worker.
        store: The `ObjectStore` to save files to.
        obstore_queue: The MPMC queue of bytes that have been downloaded, and their destination
            paths on object storage. This is the input to the obstore workers.
        max_retries: The maximum number of times to try saving each file before giving up.
            max_retries must be >= 1.
    """
    worker_id_str: str = f"obstore_worker {worker_id}:"
    log.info("%s Obstore worker starting up.", worker_id_str)
    if max_retries < 1:
        raise ValueError(f"max_retries must be > 0, not {max_retries}!")

    while True:
        try:
            obstore_file: _ObstoreFile = await obstore_queue.get()
        except asyncio.QueueShutDown:
            log.info("%s obstore_queue has shut down. Worker exiting.", worker_id_str)
            break

        dst_path = obstore_file.ftp_file.dst_obstore_path
        try:
            await store.put_async(dst_path, obstore_file.data)
        except Exception as e:
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
                await asyncio.sleep(5)
                await obstore_queue.put(obstore_file)
            else:
                log.exception(
                    "%s Giving up writing %s after %d retries. Exception: %s",
                    worker_id_str,
                    dst_path,
                    max_retries,
                    e,
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
