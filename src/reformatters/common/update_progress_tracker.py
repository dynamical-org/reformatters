import json
import queue
import threading
from collections.abc import Sequence

import fsspec

from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry
from reformatters.common.storage import StoreFactory

log = get_logger(__name__)

PROCESSED_VARIABLES_KEY = "processed_variables"


class UpdateProgressTracker:
    """
    Tracks which variables have been processed within a time slice of a job.
    Allows for skipping already processed variables in case the process is interrupted.
    """

    def __init__(
        self,
        reformat_job_name: str,
        time_i_slice_start: int,
        store_factory: StoreFactory,
    ) -> None:
        self.reformat_job_name = reformat_job_name
        self.time_i_slice_start = time_i_slice_start
        self.queue: queue.Queue[str] = queue.Queue()

        self.fs, relative_store_path = store_factory.primary_store_fsspec_filesystem()
        self.update_progress_dir = relative_store_path.replace(
            ".zarr", "_update_progress"
        )

        if isinstance(self.fs, fsspec.implementations.local.LocalFileSystem):
            self.fs.makedirs(self.update_progress_dir, exist_ok=True)

        try:
            file_content = retry(
                lambda: self.fs.read_text(self._get_path(), encoding="utf-8"),
                max_attempts=1,
            )
            self.processed_variables: set[str] = set(
                json.loads(file_content)[PROCESSED_VARIABLES_KEY]
            )
            log.info(
                f"Loaded {len(self.processed_variables)} processed variables: {self.processed_variables}"
            )
        except FileNotFoundError:
            self.processed_variables = set()

        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def record_completion(self, var: str) -> None:
        self.queue.put(var)

    def get_unprocessed[T: BaseInternalAttrs](
        self, all_vars: Sequence[DataVar[T]]
    ) -> list[DataVar[T]]:
        # Edge case: if all variables have been processed, but the job failed on writing metadata,
        # reprocess (any) one variable to ensure metadata is written.
        unprocessed = [v for v in all_vars if v.name not in self.processed_variables]
        if len(unprocessed) == 0:
            return [all_vars[0]]
        return unprocessed

    def close(self) -> None:
        try:
            retry(lambda: self.fs.rm(self._get_path()), max_attempts=1)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Could not delete progress file: {e}")

    def _get_path(self) -> str:
        return f"{self.update_progress_dir}/_internal_update_progress_{self.reformat_job_name}_{self.time_i_slice_start}.json"

    def _process_queue(self) -> None:
        """Run as a background thread to process variables from the queue and record progress."""
        while True:
            try:
                var = self.queue.get()
                self.processed_variables.add(var)

                def _write_content() -> None:
                    content = json.dumps(
                        {PROCESSED_VARIABLES_KEY: list(self.processed_variables)}
                    )
                    self.fs.pipe(self._get_path(), content.encode("utf-8"))

                retry(
                    _write_content,
                    max_attempts=3,
                )

                self.queue.task_done()
            except Exception as e:  # noqa: BLE001
                log.warning(f"Could not record progress for variable {e}")
