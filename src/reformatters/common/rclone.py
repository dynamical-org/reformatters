"""Shared subprocess, listing and stats-logging helpers for `rclone`-based archivers.
See `reformatters.dwd.archive_gribs`, `reformatters.eccc.hrdps.archive_gribs` and
`reformatters.ecmwf.archive_gribs`."""

import os
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from subprocess import PIPE
from typing import IO, Any, Final

from reformatters.common.logging import get_logger

log = get_logger(__name__)

RCLONE: Final[str] = "/usr/bin/rclone"


def _child_env(env_vars: dict[str, Any] | None) -> dict[str, Any] | None:
    """Add `env_vars` to this process's environment, which `rclone` needs for HOME, PATH and proxies."""
    if not env_vars:
        return None
    return os.environ | env_vars


def run_command_with_concurrent_logging(
    cmd: Sequence[str],
    env_vars: dict[str, Any] | None = None,
) -> int:
    cmd_str = " ".join(cmd)
    log.info("Running command: %s", cmd_str)

    process = None
    try:
        process = subprocess.Popen(  # noqa: S603
            cmd,
            text=True,
            stdout=PIPE,
            stderr=PIPE,
            bufsize=1,
            env=_child_env(env_vars),
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


def list_files(
    path: str,
    checkers: int,
    rclone_args: Sequence[str] = (),
    env_vars: dict[str, Any] | None = None,
    timeout_seconds: float = 200,
) -> list[PurePosixPath]:
    """List files recursively.

    Uses `rclone lsf` (list files) command: https://rclone.org/commands/rclone_lsf

    The returned paths do not include the input `path`. For example, if there's just 1 file on disk:
    "/foo/bar/baz.qux", and `list_files` is called with `path="/foo/"` then the returned path will
    be "bar/baz.qux".

    Args:
        path: List all the files in this path recursively. This must be in the form that `rclone`
            expects, such as `remote:path` (e.g. `dwd-http:/weather/nwp/icon-eu-grib/00/`) or, for a
            path on a local file system, just use the absolute path.
        checkers: This number is passed to the `rclone --checkers` argument.
            In the context of recursive file listing, it appears `checkers` controls the number of
            directories that are listed in parallel. Note that more is not always better. For
            example, on a small VM with only 2 CPUs, `rclone` maxes out the CPUs if `checkers` is
            above 32, and this actually slows down file listing.
            For more info, see the rclone docs: https://rclone.org/docs/#checkers-int
        rclone_args: Additional args to be passed to `rclone lsf`.
        env_vars: Environment variables to add to this process's environment for `rclone`.
        timeout_seconds: Kill the `rclone` subprocess if it runs longer than this. Scoped to a
            single init directory, so list operation latency (not total bucket size) sets the
            duration; the default is generous because the pod deadline is hours.

    Returns:
        paths: A sorted list of all the files found in `path`. Returns an empty list if the
        directory does not exist.
    """
    log.info("Listing files on '%s'...", path)
    cmd = (
        RCLONE,
        "lsf",
        path,
        "--fast-list",
        "--recursive",
        "--files-only",
        f"--checkers={checkers:d}",
        *rclone_args,
    )
    log.info("Running command: '%s'", " ".join(cmd))
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            text=True,
            capture_output=True,
            env=_child_env(env_vars),
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as e:
        if (
            e.returncode == 3
            and isinstance(e.stderr, str)
            and "directory not found" in e.stderr.lower()
        ):
            log.info("Directory not found: '%s'", path)
            return []
        else:
            log.exception(
                "stderr: %s; stdout: %s",
                _convert_called_process_error_output_to_str(e.stderr),
                _convert_called_process_error_output_to_str(e.stdout),
            )
            raise
    else:
        if result.stderr:
            log.info("rclone stderr: %s", result.stderr)
        paths = sorted(PurePosixPath(p) for p in result.stdout.splitlines())
        log.info(f"Found {len(paths):,d} files on '{path}'.")
        return paths


def copy_local_file(
    src_path: Path,
    dst_path: str,
    stats_logging_freq: str = "1m",
    env_vars: dict[str, Any] | None = None,
) -> None:
    """Copy one local file to `dst_path`, which must be in the form `rclone` expects."""
    cmd = (
        RCLONE,
        "copyto",
        str(src_path),
        dst_path,
        "--s3-no-check-bucket",  # Workaround for reformatters issue #428
        f"--stats={stats_logging_freq}",
        "--stats-log-level=ERROR",  # Output stats to stderr.
        "--quiet",  # Only output logs at error level.
        "--stats-one-line",
    )
    return_code = run_command_with_concurrent_logging(cmd, env_vars=env_vars)
    if return_code != 0:
        raise RuntimeError(
            f"rclone copyto exited with code {return_code} for '{src_path}' -> '{dst_path}'"
        )


def _convert_called_process_error_output_to_str(output: str | bytes | None) -> str:
    if output is None:
        return ""
    elif isinstance(output, str):
        return output
    else:
        return output.decode()
