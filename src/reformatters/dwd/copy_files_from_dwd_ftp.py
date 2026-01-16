"""
Restructure DWD GRIB files from FTP to a timestamped directory structure using `rclone`.

`rclone` is a command-line application to manage files on many different storage systems, including
FTP, cloud object storage, and the local filesystem. See https://rclone.org


## EXAMPLE OF DESIRED TRANSFORMATION:

Example source path (DWD's directory structure for ICON-EU as of January 2026):

    /weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2


Another example source path (DWD's directory structure for ICON-D2-RUC as of Jan 2026, which might
become the directory structure for ICON-EU in the future[1]):

    /weather/nwp/v1/m/icon-d2-ruc/p/T_2M/r/2026-01-14T02:00/s/PT000H00M.grib2


Example destination path:

    /2026-01-14T00Z/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2


## WHY DOES THIS PYTHON FILE EXIST? CAN'T WE JUST USE `rclone copy`?

`rclone copy --name-transform` cannot restructure the directory based on the timestamp in the
filename because:

1. rclone processes path segments individually, one by one, from left-to-right.
2. rclone explicitly prohibits adding path separators (`/`) during a name transformation.

For example, when processing the path `00/alb_rad/icon-eu_2026012300.grib2.bz2`, `rclone` will
process '00' first, before it has access to the datetime in the leaf filename. `rclone` cannot
rename one part of the path based on a subsequent part of the path. This fundamental limitation
persists, no matter if we use `regex=` or `command=` with `rclone copy --name-transform`.

Consequently, rclone cannot dynamically create new directory levels based on filename content.
Instead, we group files by their NWP init time and NWP variable name in Python and call `rclone copy
--files-from-raw` for each of these groups of files.


## IF WE NEED MORE PERFORMANCE

This code calls `rclone copy --files-from-raw` once per NWP variable. This wastes a bit of time
because `rclone` has to re-establish its pool of FTP connections for each NWP variable. This wastes
maybe 0.5 seconds to 1 second for each NWP variable. There are two ways to get more performance:

1. Change `_compute_copy_plan` so it groups files _only_ by NWP init datetime. This will allow us to
   send much larger batches to `rclone` in one go (But this won't work for DWD's new directory scheme.
   See details in the docstring for `_compute_copy_plan`.)
2. Call `rclone rc` (where rc is short for remote control). This runs `rclone` as a daemon which
   listens for commands. This will make our code a little more complex. But it would allow `rclone`
   to keep its FTP clients alive for the duration of the entire transfer.


## REFERENCES

1. https://www.dwd.de/DE/leistungen/opendata/neuigkeiten/opendata_april2025_1.html
"""

import csv
import ctypes
import io
import logging
import re
import signal
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Final, NamedTuple

from reformatters.common.logging import get_logger

log = get_logger(__name__)

GIBIBYTE: Final[int] = 1024**3

# Constants for Linux prctl (Process Control)
PR_SET_PDEATHSIG: Final[int] = 1


class _PathAndSize(NamedTuple):
    path: PurePosixPath
    size_bytes: int


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
                  This is useful for testing locally.
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
) -> list[_PathAndSize]:
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
        "--format=ps",  # Return the path and size.
        "--csv",  # Separate path and size with a comma, and escape any commas.
        "--config=",  # There is no config file because we pass everything as command-line args.
    ]

    stdout_str = _run_rclone(cmd, timeout=timedelta(seconds=90))
    file_list: list[_PathAndSize] = []
    reader = csv.reader(io.StringIO(stdout_str))
    file_list = [
        _PathAndSize(path=PurePosixPath(row[0]), size_bytes=int(row[1]))
        for row in reader
        if row
    ]
    total_size_gibibytes = sum(f.size_bytes for f in file_list) / GIBIBYTE
    log.info(
        f"Found {len(file_list):,d} files (totalling {total_size_gibibytes:.3f} GiB)"
        f" in {ftp_url} (before any filtering)"
    )
    return sorted(file_list, key=lambda x: x.path)


def _compute_copy_plan(
    file_list: list[_PathAndSize],
    max_files_per_nwp_variable: int = sys.maxsize,
) -> dict[tuple[datetime, str], list[_PathAndSize]]:
    """Groups files by their NWP initialization datetime and variable name.

    Returns dict[(nwp_init_datetime, nwp_variable_name)] = list[file_path_and_size].
    Where `file_path` starts with (and includes) the NWP variable name.

    ## Implementation note:

    While DWD continue to use their "legacy" directory structure [1] we _could_ group filenames by
    _only_ the NWP initialisation datetime (instead of grouping by init time _and_ variable name).
    We'd then give `rclone` one huge list of all the files below the init hour. This would work
    because, below the init hour, the paths are both of the form variable_name/filename.grib2.bz2.

    There are two main reasons that we group by both the init datetime _and_ variable name:

    1. Crucially, grouping by _just_ the init datetime will break if/when DWD move to their new
       directory structure [2].
    2. Having smaller groups gives us more control and visibility into what `rclone` is doing. This
       should help with debugging. (Although, if we _really_ wanted to leave `rclone` running for a
       long time, we could stream `stderr` and `stdout` into the Python logger.)

    ## Footnotes:

    1. DWD's legacy directory structure:
       /weather/nwp/icon-eu/grib/00/
           alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2
    2. DWD's new directory structure (already used for ICON-D2-RUC):
       /weather/nwp/v1/m/icon-d2-ruc/p/T_2M/r/2026-01-14T02:00/s/PT000H00M.grib2
    """
    copy_plan: dict[tuple[datetime, str], list[_PathAndSize]] = defaultdict(list)
    date_regex = re.compile(r"_(\d{10})_")
    n_expected_path_parts: Final[int] = 2

    for file_to_be_copied in file_list:
        if "pressure-level" in file_to_be_copied.path.name:
            continue

        if len(file_to_be_copied.path.parts) != n_expected_path_parts:
            log.warning("Unexpected path structure: %s", file_to_be_copied.path)
            continue

        match = date_regex.search(file_to_be_copied.path.name)
        if not match:
            log.warning("Skipping file (no date found): %s", file_to_be_copied.path)
            continue

        timestamp_str = match.group(1)
        nwp_init_datetime = datetime.strptime(timestamp_str, "%Y%m%d%H")
        nwp_variable_name = file_to_be_copied.path.parts[0]
        key = (nwp_init_datetime, nwp_variable_name)

        n_files_for_nwp_var_and_init = len(copy_plan[key])
        if n_files_for_nwp_var_and_init < max_files_per_nwp_variable:
            copy_plan[key].append(file_to_be_copied)

    return copy_plan


def _copy_batches(
    ftp_host: str,
    ftp_path: PurePosixPath,
    dst_root: PurePosixPath,
    copy_plan: dict[tuple[datetime, str], list[_PathAndSize]],
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
    files_to_be_copied: list[_PathAndSize],
    transfers: int,
) -> None:
    """Executes a single rclone copy batch for a specific destination.

    Args:
        ftp_host: The FTP host (e.g. 'opendata.dwd.de').
        ftp_path: The root source path on the FTP server (including NWP init hour).
        dst_path: The specific destination directory (including NWP init timestamp).
        files_to_be_copied: List of file path and sizes relative to ftp_path to be copied.
        transfers: Number of parallel transfers to use for this batch.
    """
    # Modern Linux platforms often install `rclone` as a sandboxed snap, which does not have access
    # to `/tmp`, to we store the temporary file in the current working directory.
    with NamedTemporaryFile(mode="w", dir=".", prefix=".rclone_files_") as list_file:
        # rel_paths are relative to ftp_path
        list_file.write("\n".join(p.path.as_posix() for p in files_to_be_copied))
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
            _run_rclone(cmd, timeout=timedelta(minutes=30))

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


def _run_rclone(cmd: list[str], timeout: timedelta) -> str:
    """Runs a command with logging and safety measures and returns stdout."""
    log.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,  # Open stdout and stderr pipes in text mode (not bytes).
            check=True,  # Raise CalledProcessError if returncode != 0.
            timeout=round(timeout.total_seconds()),
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
