import json
import queue
import threading
from collections.abc import Sequence

import zarr.storage

from reformatters.common.config_models import BaseInternalAttrs, DataVar
from reformatters.common.fsspec import fsspec_apply
from reformatters.common.logging import get_logger
from reformatters.common.zarr import _get_fs_and_path

log = get_logger(__name__)

PROCESSED_VARIABLES_KEY = "processed_variables"


class UpdateProgressTracker:
    """
    Tracks which variables have been processed within a time slice of a job.
    Allows for skipping already processed variables in case the process is interrupted.
    """

    def __init__(
        self,
        store: zarr.abc.store.Store,
        job_name: str,
        time_i_slice_start: int,
    ):
        self.job_name = job_name
        self.time_i_slice_start = time_i_slice_start
        self.queue: queue.Queue[str] = queue.Queue()
        self.fs, self.path = _get_fs_and_path(store)

        try:
            file_content = fsspec_apply(
                self.fs, "cat_file", self._get_path(), max_attempts=1
            )
            self.processed_variables: set[str] = set(
                json.loads(file_content.decode("utf-8"))[PROCESSED_VARIABLES_KEY]
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

    def get_unprocessed_str(self, all_vars: Sequence[str]) -> list[str]:
        # Method used by pre-RegionJob reformatters.
        unprocessed = [v for v in all_vars if v not in self.processed_variables]
        # Edge case: if all variables have been processed, but the job failed on writing metadata,
        # reprocess (any) one variable to ensure metadata is written.
        if len(unprocessed) == 0:
            return [all_vars[0]]
        return unprocessed

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
            fsspec_apply(self.fs, "rm", self._get_path(), max_attempts=1)
        except Exception as e:
            log.warning(f"Could not delete progress file: {e}")

    def _get_path(self) -> str:
        return f"{self.path}/_internal_update_progress_{self.job_name}_{self.time_i_slice_start}.json"

    def _process_queue(self) -> None:
        """Run as a background thread to process variables from the queue and record progress."""
        while True:
            try:
                var = self.queue.get()
                self.processed_variables.add(var)

                content = json.dumps(
                    {PROCESSED_VARIABLES_KEY: list(self.processed_variables)}
                )
                fsspec_apply(self.fs, "pipe", self._get_path(), content.encode("utf-8"))

                self.queue.task_done()
            except Exception as e:
                log.warning(f"Could not record progress for variable {e}")
