import json
import logging
from typing import Any, Final, NamedTuple

from reformatters.common.logging import get_logger

log = get_logger(__name__)

_MIBIBYTE: Final[int] = 1024**2
_GIBIBYTE: Final[int] = 1024**3


class TransferSummary(NamedTuple):
    """A subset of the values returned by rclone at the end of a copy operation.

    rclone docs:
    - https://rclone.org/rc/#core-stats
    """

    total_transfers: int = 0  # number of files transferred
    total_bytes: int = 0
    total_checks: int = 0  # number of files checked by rclone
    errors: int = 0  # number of errors
    elapsed_time: float = 0  # seconds since rclone started
    transfer_time: float = 0  # seconds spent running jobs
    listed: int = 0  # number of directory entries listed

    @classmethod
    def from_rclone_stats(cls, log_entries: list[dict[str, Any]]) -> "TransferSummary":
        """Extracts the final TransferSummary from rclone log entries."""
        # Find the last JSON line that contains a "stats" key
        stats = {}
        for entry in reversed(log_entries):
            if "stats" in entry:
                stats = entry["stats"]
                break
        else:
            raise ValueError("No stats in log_entries!")

        return TransferSummary(
            total_transfers=stats["totalTransfers"],
            total_bytes=stats["totalBytes"],
            total_checks=stats["totalChecks"],
            errors=stats["errors"],
            elapsed_time=stats["elapsedTime"],
            transfer_time=stats["transferTime"],
            listed=stats["listed"],
        )

    def __add__(self, other: object) -> "TransferSummary":
        if not isinstance(other, TransferSummary):
            return NotImplemented
        return TransferSummary(
            total_transfers=self.total_transfers + other.total_transfers,
            total_bytes=self.total_bytes + other.total_bytes,
            total_checks=self.total_checks + other.total_checks,
            errors=self.errors + other.errors,
            elapsed_time=self.elapsed_time + other.elapsed_time,
            transfer_time=self.transfer_time + other.transfer_time,
            listed=self.listed + other.listed,
        )

    def __str__(self) -> str:
        bytes_per_sec = self.total_bytes / self.transfer_time
        mibibytes_per_sec = bytes_per_sec / _MIBIBYTE
        return (
            f"{self.total_transfers} files transferred, "
            f"{mibibytes_per_sec:.3f} MiB/sec, "
            f"{format_bytes(self.total_bytes)} total transferred, "
            f"{self.total_checks} files checked, "
            f"{self.errors} errors, "
            f"{self.elapsed_time} seconds rsync runtime, "
            f"{self.transfer_time} seconds transfer time, "
            f"{self.listed} number of directories listed."
        )


def parse_and_log_rclone_json(stderr: str) -> list[dict[str, Any]]:
    """Parses rclone stderr and logs with appropriate levels."""
    # See https://rclone.org/docs/#use-json-log for rclone's JSON schema.
    log_entries: list[dict[str, Any]] = []
    for line in stderr.splitlines():
        rclone_log_entry = json.loads(line)
        log_entries.append(rclone_log_entry)

        rclone_log_level = rclone_log_entry["level"]
        if "stats" in rclone_log_entry and rclone_log_level == "info":
            continue

        # Map rclone log levels to Python logging levels
        python_log_level = logging.getLevelNamesMapping()[rclone_log_level.upper()]
        if python_log_level == logging.INFO and "object" in rclone_log_entry:
            # Demote per-file logs for successful transfers.
            python_log_level = logging.DEBUG

        msg = rclone_log_entry["msg"]
        log.log(python_log_level, "rclone: %s", msg)

    return log_entries


def format_bytes(size_bytes: int) -> str:
    size_gibibytes = size_bytes / _GIBIBYTE
    return f"{size_gibibytes:.3f} GiB"
