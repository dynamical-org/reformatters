"""A simple tool to copy files from an FTP server to anywhere that `obstore`
can write to.

Single FTP connections are slow. So this code uses multiple concurrent FTP connections to maximise
throughput. Maximising throughput for a single virtual machine should minimise compute costs.

## Overview of architecture

Instead of manually programming two threadpools, the code uses an `asyncio.TaskGroup` containing
a set of `ftp_worker`s and a set of `obstore_worker`s.

The code uses two MPMC queues. The first queue passes `FtpTask`s to the `ftp_worker`s. The second
queue passes the data payload to the `obstore_worker`s.

Each `ftp_worker` keeps an `aioftp.Client` alive until there are no more `FtpTask`s.
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
class FtpFile:
    src_ftp_path: PurePosixPath
    dst_obstore_path: str
    n_retries: int = field(default=0, init=False)
    ftp_errors: list[Exception] = field(default_factory=list, init=False)


async def copy_files_from_ftp_to_obstore(
    ftp_host: str,
    files: Iterable[FtpFile],
    dst_store: ObjectStore,
    n_ftp_workers: int = 8,
    n_obstore_workers: int = 8,
) -> None:
    """Copy files from an FTP server to obstore."""
    # Copy FtpFile objects to the ftp_queue:
    ftp_queue: Queue[FtpFile] = Queue()
    for ftp_file in files:
        ftp_queue.put_nowait(ftp_file)

    # Initialise other queues:
    failed_ftp_tasks: Queue[FtpFile] = Queue()

    # Set maxsize of Queue to apply back-pressure to the FTP workers.
    obstore_queue: Queue[_ObstoreFile] = Queue(maxsize=n_obstore_workers * 2)

    # --- Start the Pipeline ---
    async with asyncio.TaskGroup() as tg:
        for worker_id in range(n_ftp_workers):
            tg.create_task(
                _ftp_worker(
                    worker_id=worker_id,
                    ftp_host=ftp_host,
                    ftp_queue=ftp_queue,
                    obstore_queue=obstore_queue,
                    failed_ftp_tasks=failed_ftp_tasks,
                )
            )

        for worker_id in range(n_obstore_workers):
            tg.create_task(_obstore_worker(worker_id, dst_store, obstore_queue))

        print("await _ftp_queue.join()")
        await ftp_queue.join()
        print("joined _ftp_queue!")
        obstore_queue.shutdown()

    if failed_ftp_tasks.empty():
        print("Good news: No failed FTP tasks!")
    else:
        while True:
            try:
                ftp_task: FtpFile = failed_ftp_tasks.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                print("Failed FTP task:", ftp_task)
                failed_ftp_tasks.task_done()


@dataclass
class _ObstoreFile:
    data: bytes
    dst_obstore_path: str
    n_retries: int = field(default=0, init=False)


async def _ftp_worker(
    worker_id: int,
    ftp_host: str,
    ftp_queue: asyncio.Queue[FtpFile],
    obstore_queue: asyncio.Queue[_ObstoreFile],
    failed_ftp_tasks: asyncio.Queue[FtpFile],
    max_retries: int = 3,
) -> None:
    """A worker that keeps a single FTP connection alive and processes queue
    items."""
    worker_id_str: str = f"ftp_worker {worker_id}:"
    print(worker_id_str, "Starting up...")

    # Establish the persistent client connection
    async with aioftp.Client.context(ftp_host) as ftp_client:
        print(worker_id_str, "Connection established and logged in.")

        # Start continuous processing loop
        while True:
            try:
                ftp_task: FtpFile = ftp_queue.get_nowait()
            except asyncio.QueueEmpty:
                print(worker_id_str, "ftp_queue is empty. Finishing.")
                break

            print(worker_id_str, f"Attempting to download {ftp_task=}")

            try:
                async with ftp_client.download_stream(ftp_task.src_ftp_path) as stream:
                    data = await stream.read()
            except aioftp.StatusCodeError as e:
                ftp_task.ftp_errors.append(e)
                _log_ftp_exception(ftp_task, e, worker_id_str)
                if ftp_task.n_retries < max_retries:
                    ftp_task.n_retries += 1
                    print(
                        worker_id_str,
                        "WARNING: Putting ftp_task back on queue to retry later.",
                    )
                    await ftp_queue.put(ftp_task)
                else:
                    print(worker_id_str, "ERROR: Giving up on ftp_task")
                    await failed_ftp_tasks.put(ftp_task)
            else:
                print(worker_id_str, f"Finished downloading {ftp_task=}")
                await obstore_queue.put(_ObstoreFile(data, ftp_task.dst_obstore_path))
            finally:
                ftp_queue.task_done()


async def _obstore_worker(
    worker_id: int,
    store: ObjectStore,
    obstore_queue: asyncio.Queue[_ObstoreFile],
) -> None:
    """Obstores are designed to work concurrently, so we can share one
    `obstore` between tasks."""
    worker_id_str: str = f"obstore_worker {worker_id}:"
    while True:
        print(worker_id_str, "Getting obstore task")
        try:
            obstore_task: _ObstoreFile = await obstore_queue.get()
        except asyncio.QueueShutDown:
            print(worker_id_str, "obstore_queue has shut down!")
            break

        await store.put_async(obstore_task.dst_obstore_path, obstore_task.data)
        obstore_queue.task_done()
        print(
            worker_id_str,
            f"Done writing to {obstore_task.dst_obstore_path} after {obstore_task.n_retries} retries.",
        )

        # TODO(Jack): Handle retries


def _log_ftp_exception(
    ftp_task: FtpFile, e: aioftp.StatusCodeError, ftp_worker_id_str: str
) -> None:
    error_str = f"{ftp_worker_id_str} WARNING: "
    # Check if the file is missing from the FTP server:
    if (
        isinstance(e, aioftp.StatusCodeError)
        and isinstance(e.received_codes, tuple)
        and len(e.received_codes) == 1
        and e.received_codes[0].matches("550")
    ):
        error_str += f"File not available on FTP server: {ftp_task.src_ftp_path} "

    print(
        error_str,
        f"ftp_client.download_stream raised exception whilst processing {ftp_task=}",
        e,
    )
