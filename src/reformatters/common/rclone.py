"""Shared subprocess + stats-logging helpers for `rclone`-based archivers.
See `reformatters.dwd.archive_gribs` and `reformatters.eccc.hrdps.archive_gribs`."""

import subprocess
import threading
from collections.abc import Sequence
from subprocess import PIPE
from typing import IO, Any, Final

from reformatters.common.logging import get_logger

log = get_logger(__name__)


def run_command_with_concurrent_logging(
    cmd: Sequence[str],
    env_vars: dict[str, Any] | None = None,
) -> int:
    cmd_str = " ".join(cmd)
    log.info("Running command: %s", cmd_str)

    process = None
    try:
        process = subprocess.Popen(  # noqa: S603
            cmd, text=True, stdout=PIPE, stderr=PIPE, bufsize=1, env=env_vars
        )

        # Create threads to read stdout and stderr simultaneously
        t1 = threading.Thread(target=_log_stdout, args=(process.stdout,))
        t2 = threading.Thread(target=_log_stderr_stats, args=(process.stderr,))

        t1.start()
        t2.start()

        # Wait for threads to finish (which happens when process closes the pipes)
        t1.join()
        t2.join()

        return_code = process.wait()
    except KeyboardInterrupt:
        # Avoid having a zombie rclone process if user kills Python with Ctrl-C
        log.warning("Received KeyboardInterrupt... terminating subprocess...")
        if process:
            process.terminate()
        raise
    else:
        log.info("return code = %d after running command: '%s'", return_code, cmd_str)
        return return_code


def _log_stdout(pipe: IO[str]) -> None:
    """Reads a pipe line-by-line and logs it."""
    with pipe:
        for line in pipe:
            log.info(f"stdout: {line.strip()}")


def _log_stderr_stats(pipe: IO[str]) -> None:
    with pipe:
        for line in pipe:
            try:
                tidy_line = _tidy_stats(line)
            except Exception:  # noqa: BLE001
                # An exception here just means the line wasn't a stats line,
                # so let's log it and move on. No biggie.
                log.info("stderr: '%s'", line)
            else:
                log.info(f"Rclone stats: {tidy_line}")


def _tidy_stats(line: str) -> str:
    """Remove meaningless (and hence confusing) numbers from rclone stats!

    Example raw stats output from rclone:

        2026/01/31 16:15:41 ERROR :    16.342 MiB / 18.818 MiB, 87%, 0 B/s, ETA -
                            ^^^^^                 ^^^^^^^^^^^^  ^^^         ^^^^^
    Issues to fix:    Stats aren't an error!      And these numbers mean nothing!
    """
    split_on: Final[str] = "ERROR :"
    if split_on not in line:
        raise ValueError(f"Expected a colon in rclone stats line: '{line}'")
    line = line.split(split_on, 1)[1]

    parts = line.split(",")
    n_expected_parts: Final[int] = 4
    if len(parts) != n_expected_parts:
        raise ValueError(
            f"Expected {n_expected_parts} comma-separated values in rclone stats line. Line: '{line}'"
        )

    transferred_bytes = parts[0].split("/")[0].strip()
    speed = parts[2].strip()
    return f"Transferred so far: {transferred_bytes}. Recent throughput: {speed}"
