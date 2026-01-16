import json
import logging
from typing import Any, NamedTuple

from reformatters.common.logging import get_logger

log = get_logger(__name__)


class TransferSummary(NamedTuple):
    transfers: int = 0
    checks: int = 0
    errors: int = 0
    bytes_transferred: int = 0


def parse_and_log_rclone_json(stderr: str, quiet: bool = False) -> list[dict[str, Any]]:
    """Parses rclone stderr and logs with appropriate levels."""
    log_entries: list[dict[str, Any]] = []
    for line in stderr.splitlines():
        if not (clean_line := line.strip()):
            continue

        log.debug(f"Full JSON: {clean_line}")

        try:
            entry = json.loads(clean_line)
        except json.JSONDecodeError:
            # Fallback for non-JSON lines
            line_upper = clean_line.upper()
            if "ERROR" in line_upper or "FAILED" in line_upper:
                log_level = logging.ERROR
            elif "WARNING" in line_upper:
                log_level = logging.WARNING
            else:
                log_level = logging.INFO

            if not (quiet and log_level == logging.INFO):
                log.log(log_level, f"rclone: {clean_line}")
            continue

        level_str = entry.get("level", "info").lower()
        msg = entry.get("msg", "")

        # Map rclone levels to Python logging levels
        if level_str == "error":
            log_level = logging.ERROR
        elif level_str == "warning":
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        entry["python_log_level"] = log_level
        log_entries.append(entry)

        # Skip per-file info logs if quiet is True
        if not (quiet and log_level == logging.INFO and "object" in entry):
            log.log(log_level, f"rclone: {msg}")

    return log_entries


def summarize_transfers(log_entries: list[dict[str, Any]]) -> TransferSummary:
    """Extracts the final TransferSummary from rclone log entries."""
    # Find the last entry with "stats"
    final_stats = {}
    for entry in reversed(log_entries):
        if "stats" in entry:
            final_stats = entry["stats"]
            break

    if not final_stats:
        return TransferSummary()

    return TransferSummary(
        transfers=final_stats.get("transfers", 0),
        checks=final_stats.get("checks", 0),
        errors=final_stats.get("errors", 0),
        bytes_transferred=final_stats.get("bytes", 0),
    )
