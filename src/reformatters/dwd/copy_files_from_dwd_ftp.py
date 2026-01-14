"""
Restructure DWD GRIB files from FTP to a timestamped directory structure using `rclone`.

`rclone` is a command-line application to manage files on many different storage systems, including
FTP, cloud object storage, and the local filesystem. See https://rclone.org

## Example of desired transformation:

Source:      00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2
Destination: 2026-01-14T00Z/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2

## Why does this Python file exist? Can't we just use `rclone copy`?

`rclone copy --name-transform` cannot restructure the directory based on the timestamp in the
filename because:

1. rclone processes path segments individually.
2. rclone explicitly prohibits adding path separators (`/`) during a name transformation.

For example, in the path `00/alb_rad/icon-eu_2026012300.grib2.bz2`, `rclone` has no access to the
datetime in the leaf filename when `rclone` is transforming the '00' part of the filename.

Consequently, rclone cannot dynamically create new directory levels based on filename content.
Instead, we group files by their timestamp in Python and call `rclone copy --files-from-raw` for
each of these groups of files.
"""

import ctypes
import logging
import re
import signal
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Final

from reformatters.common.logging import get_logger

log = get_logger(__name__)

# Constants for Linux prctl (Process Control)
PR_SET_PDEATHSIG = 1


def copy_files_from_dwd_ftp(
    ftp_host: str,
    ftp_path: PurePosixPath,
    dst_root: PurePosixPath,
    transfers: int = 10,
    max_files_per_nwp_variable: int = sys.maxsize,
) -> None:
    """Restructure DWD GRIB files from FTP to a timestamped directory structure.

    Args:
        ftp_host: The FTP host, e.g. 'opendata.dwd.de'
        ftp_path: The source path on the FTP host, e.g. '/weather/nwp/icon-eu/grib/00'
        dst_root: The destination root directory.
        transfers: Number of parallel transfers. DWD appears to limit the number of parallel
                   transfers from one IP address to about 10.
        max_files_per_nwp_variable: Optional limit on the number of files to transfer per NWP variable.
    """
    file_list = list_ftp_files(ftp_host=ftp_host, ftp_path=ftp_path)
    copy_plan = _compute_copy_plan(
        file_list=file_list,
        max_files_per_nwp_variable=max_files_per_nwp_variable,
    )
    _copy_batches(
        ftp_host=ftp_host,
        ftp_path=ftp_path,
        dst_root=dst_root,
        copy_plan=copy_plan,
        transfers=transfers,
    )


def list_ftp_files(
    ftp_host: str,
    ftp_path: PurePosixPath,
    timeout: int = 90,
) -> list[PurePosixPath]:
    """Recursively list all files below ftp_host/ftp_path."""
    ftp_url = f"ftp://{ftp_host}{ftp_path}"
    log.info("Listing %s...", ftp_url)
    cmd = [
        "rclone",
        "lsf",
        "--recursive",
        "--files-only",
        f":ftp:{ftp_path}",
        *_get_rclone_ftp_args(ftp_host),
        "--config=",  # There is no config file because we pass everything as command-line args.
    ]

    stdout_str = _run_rclone(cmd, timeout=timeout)
    file_list = [
        PurePosixPath(line.strip()) for line in stdout_str.splitlines() if line.strip()
    ]
    log.info(f"Found {len(file_list):,d} files in {ftp_url}")
    return sorted(file_list)


def _compute_copy_plan(
    file_list: list[PurePosixPath],
    max_files_per_nwp_variable: int = sys.maxsize,
) -> dict[tuple[datetime, str], list[PurePosixPath]]:
    """Groups files by their NWP initialization datetime and variable name.

    Returns dict[(nwp_init_datetime, nwp_variable_name)] = list[file_path].
    Where `file_path` starts with (and includes) the NWP variable name.
    """
    copy_plan: dict[tuple[datetime, str], list[PurePosixPath]] = defaultdict(list)
    date_regex = re.compile(r"_(\d{10})_")
    n_expected_path_parts: Final[int] = 2

    for file_to_be_copied in file_list:
        if "pressure-level" in file_to_be_copied.name:
            continue

        if len(file_to_be_copied.parts) != n_expected_path_parts:
            log.warning("Unexpected path structure: %s", file_to_be_copied)
            continue

        match = date_regex.search(file_to_be_copied.name)
        if not match:
            log.warning("Skipping file (no date found): %s", file_to_be_copied)
            continue

        timestamp_str = match.group(1)
        nwp_init_datetime = datetime.strptime(timestamp_str, "%Y%m%d%H")
        nwp_variable_name = file_to_be_copied.parts[0]
        key = (nwp_init_datetime, nwp_variable_name)

        n_files_for_nwp_var_and_init = len(copy_plan[key])
        if n_files_for_nwp_var_and_init < max_files_per_nwp_variable:
            copy_plan[key].append(file_to_be_copied)

    return copy_plan


def _copy_batches(
    ftp_host: str,
    ftp_path: PurePosixPath,
    dst_root: PurePosixPath,
    copy_plan: dict[tuple[datetime, str], list[PurePosixPath]],
    transfers: int = 10,
) -> None:
    """Executes rclone copy for each timestamp and variable batch in the plan."""
    n_batches = len(copy_plan)
    for i, ((nwp_init_dt, nwp_var), files_to_be_copied) in enumerate(copy_plan.items()):
        nwp_init_datetime_str = nwp_init_dt.strftime("%Y-%m-%dT%HZ")
        dst_path = dst_root / nwp_init_datetime_str
        log.info(
            "Batch [%d/%d]: Asking rclone to copy %d file(s) to %s (if they don't already exist)...",
            i + 1,
            n_batches,
            len(files_to_be_copied),
            dst_path / nwp_var,
        )
        _copy_batch(
            ftp_host=ftp_host,
            ftp_path=ftp_path,
            dst_path=dst_path,
            files_to_be_copied=files_to_be_copied,
            transfers=transfers,
        )


def _copy_batch(
    ftp_host: str,
    ftp_path: PurePosixPath,
    dst_path: PurePosixPath,
    files_to_be_copied: list[PurePosixPath],
    transfers: int,
) -> None:
    """Executes a single rclone copy batch for a specific destination.

    Args:
        ftp_host: The FTP host (e.g. 'opendata.dwd.de').
        ftp_path: The root source path on the FTP server (including NWP init hour).
        dst_path: The specific destination directory (including NWP init timestamp).
        files_to_be_copied: List of file paths relative to ftp_path to be copied.
        transfers: Number of parallel transfers to use for this batch.
    """
    # Modern Linux platforms often install `rclone` as a sandboxed snap, which does not have access
    # to `/tmp`, to we store the temporary file in the current working directory.
    with NamedTemporaryFile(mode="w", dir=".", prefix=".rclone_files_") as list_file:
        # rel_paths are relative to ftp_path
        list_file.write("\n".join(p.as_posix() for p in files_to_be_copied))
        list_file.flush()

        cmd = [
            "rclone",
            "copy",
            f":ftp:{ftp_path}",
            str(dst_path),
            "--files-from-raw=" + list_file.name,
            f"--transfers={transfers}",
            *_get_rclone_ftp_args(ftp_host),
            "--config=",
            "--no-check-certificate",
            "--ignore-checksum",
            "--ignore-existing",
        ]

        try:
            _run_rclone(cmd, timeout=5 * 60)
        except subprocess.CalledProcessError:
            log.exception("Failed to copy batch to %s", dst_path)


def _get_rclone_ftp_args(ftp_host: str) -> list[str]:
    return [
        "--ftp-host=" + ftp_host,
        "--ftp-user=anonymous",
        # rclone requires passwords to be obscured by encrypting & encoding them in base64.
        # The base64 string below was created with the command `rclone obscure guest`.
        "--ftp-pass=JUznDm8DV5bQBCnXNVtpK3dN1qHB",
    ]


def _run_rclone(cmd: list[str], timeout: int) -> str:
    """Runs a command with logging and safety measures and returns stdout."""
    log.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,  # Open stdout and stderr pipes in text mode (not bytes).
            check=True,  # Raise CalledProcessError if returncode != 0.
            timeout=timeout,
            preexec_fn=_set_death_signal,
        )
    except subprocess.CalledProcessError as e:
        _log_rclone_stderr(e.stderr)
        raise
    except subprocess.TimeoutExpired as e:
        if e.stderr:
            _log_rclone_stderr(
                e.stderr if isinstance(e.stderr, str) else e.stderr.decode()
            )
        raise

    _log_rclone_stderr(result.stderr)
    return result.stdout


def _log_rclone_stderr(stderr: str) -> None:
    """Parses rclone stderr and logs with appropriate levels."""
    for line in stderr.splitlines():
        if not (clean_line := line.strip()):
            continue
        line_upper = clean_line.upper()
        if "ERROR" in line_upper or "FAILED" in line_upper:
            log_level = logging.ERROR
        elif "WARNING" in line_upper:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        log.log(log_level, f"rclone: {clean_line}")


def _set_death_signal() -> None:
    """Linux-specific: Ensure the child process dies if the parent dies."""
    libc = ctypes.CDLL("libc.so.6")
    # Send SIGTERM to the child if the parent terminates.
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
