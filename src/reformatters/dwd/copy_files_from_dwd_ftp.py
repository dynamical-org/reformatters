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
import os
import re
import signal
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any, Final, NamedTuple

from reformatters.common.logging import get_logger
from reformatters.dwd.parse_rclone_log import (
    TransferSummary,
    format_bytes,
    parse_and_log_rclone_json,
)

log = get_logger(__name__)


class _PathAndSize(NamedTuple):
    path: PurePosixPath
    size_bytes: int


def copy_files_from_dwd_ftp(
    ftp_host: str,
    ftp_path: PurePosixPath,
    dst_root: PurePosixPath,
    transfers: int = 10,
    max_files_per_nwp_variable: int = sys.maxsize,
    env_vars: dict[str, Any] | None = None,
) -> TransferSummary:
    """Restructure DWD GRIB files from FTP to a timestamped directory structure.

    Args:
        ftp_host: The FTP host, e.g. 'opendata.dwd.de'
        ftp_path: The source path on the FTP host, e.g. '/weather/nwp/icon-eu/grib/00'
        dst_root: The destination root directory. e.g. for S3, the dst_root could be: 's3:bucket/foo/bar'
        transfers: Number of parallel transfers. DWD appears to limit the number of parallel
                   transfers from one IP address to about 10.
        max_files_per_nwp_variable: Optional limit on the number of files to transfer per NWP variable.
                  This is useful for testing locally.
        env_vars: Additional environment variables to give to `rclone`. For example:
            {
                "RCLONE_S3_ENV_AUTH": True,
                "RCLONE_S3_ACCESS_KEY_ID": "key",
                "RCLONE_S3_SECRET_ACCESS_KEY": "secret",
            }
    """
    file_list = list_ftp_files(ftp_host=ftp_host, ftp_path=ftp_path)
    copy_plan = _compute_copy_plan(
        file_list=file_list,
        max_files_per_nwp_variable=max_files_per_nwp_variable,
    )
    return _copy_batches(
        ftp_host=ftp_host,
        ftp_path=ftp_path,
        dst_root=dst_root,
        copy_plan=copy_plan,
        transfers=transfers,
        env_vars=env_vars,
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
        "--format=ps",  # Return the path and size to stdout.
        "--csv",  # Separate path and size with a comma, and escape any commas in path names.
        *_get_rclone_ftp_args(ftp_host),
        *_get_common_rclone_args(),
    ]

    stdout_str, _ = _run_rclone(cmd, timeout=timedelta(seconds=90))
    file_list = _parse_rclone_list_csv(stdout_str)
    log.info(
        f"Before filtering: {len(file_list):,d} files,"
        f" totalling {format_bytes(_sum_bytes(file_list))}, found in {ftp_url}"
    )
    return sorted(file_list, key=lambda x: x.path)


def _parse_rclone_list_csv(rclone_csv: str) -> list[_PathAndSize]:
    # Parse rclone's listing as a CSV
    reader = csv.reader(io.StringIO(rclone_csv))
    file_list = [
        _PathAndSize(path=PurePosixPath(row[0]), size_bytes=int(row[1]))
        for row in reader
        if row
    ]
    return file_list


def _sum_bytes(file_list: list[_PathAndSize]) -> int:
    return sum(f.size_bytes for f in file_list)


def _compute_copy_plan(
    file_list: list[_PathAndSize],
    max_files_per_nwp_variable: int = sys.maxsize,
) -> dict[tuple[datetime, str], list[_PathAndSize]]:
    """Groups files by their NWP initialization datetime and variable name.

    Returns dict[(nwp_init_datetime, nwp_variable_name)] = list[src_file_path_and_size].

    ## Implementation note:

    While DWD continue to use their "legacy" directory structure[1] we _could_ group filenames by
    _only_ the NWP initialisation datetime (instead of grouping by init time _and_ variable name).
    We'd then give `rclone` one huge list of all the files below the init hour. This would work
    because, below the init hour, the source and destination paths are both of the form
    variable_name/filename.grib2.bz2.

    There are two main reasons that we group by both the init datetime _and_ variable name:

    1. Grouping by _just_ the init datetime will break when DWD move to their new directory structure[2].
    2. Having smaller groups gives us more control and visibility into what `rclone` is doing. This
       should help with debugging. (Although, if we _really_ wanted to leave `rclone` running for a
       long time, we could stream `stderr` and `stdout` into the Python logger.)

    ## Footnotes:

    1. DWD's legacy directory structure looks like this:
           /weather/nwp/icon-eu/grib/00/
           alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2
    2. DWD's new directory structure looks like this, and is already used for ICON-D2-RUC:
           /weather/nwp/v1/m/icon-d2-ruc/p/T_2M/r/2026-01-14T02:00/s/PT000H00M.grib2
    """
    copy_plan: dict[tuple[datetime, str], list[_PathAndSize]] = defaultdict(list)
    date_regex = re.compile(r"_(\d{10})_")
    n_expected_path_parts: Final[int] = 2

    total_bytes_after_filtering = 0
    total_files_after_filtering = 0
    for file_to_be_copied in file_list:
        if "pressure-level" in file_to_be_copied.path.name:
            continue

        if len(file_to_be_copied.path.parts) != n_expected_path_parts:
            log.warning("Unexpected path structure: %s", file_to_be_copied.path)
            continue

        match = date_regex.findall(file_to_be_copied.path.name)
        if len(match) == 0:
            log.warning("Skipping file (no date found): %s", file_to_be_copied.path)
            continue
        elif len(match) > 1:
            log.warning(
                "Expected exactly one 10-digit number in the filename (the NWP init date"
                " represented as YYYYMMDDHH), but instead found %d 10-digit numbers in path %s",
                len(match),
                file_to_be_copied.path,
            )
            continue

        timestamp_str = match[0]
        nwp_init_datetime = datetime.strptime(timestamp_str, "%Y%m%d%H")
        nwp_variable_name = file_to_be_copied.path.parts[0]
        key = (nwp_init_datetime, nwp_variable_name)

        n_files_for_nwp_var_and_init = len(copy_plan[key])
        if n_files_for_nwp_var_and_init < max_files_per_nwp_variable:
            copy_plan[key].append(file_to_be_copied)
            total_bytes_after_filtering += file_to_be_copied.size_bytes
            total_files_after_filtering += 1

    log.info(
        f"After filtering: {total_files_after_filtering:,d} files,"
        f" totalling {format_bytes(total_bytes_after_filtering)},"
        f" grouped into {len(copy_plan):d} NWP variables."
    )
    return copy_plan


def _copy_batches(
    ftp_host: str,
    ftp_path: PurePosixPath,
    dst_root: PurePosixPath,
    copy_plan: dict[tuple[datetime, str], list[_PathAndSize]],
    transfers: int = 10,
    env_vars: dict[str, Any] | None = None,
) -> TransferSummary:
    """Executes rclone copy for each timestamp and variable batch in the plan."""
    n_batches = len(copy_plan)
    total_summary = TransferSummary()
    for i, ((nwp_init_dt, nwp_var), files_to_be_copied) in enumerate(copy_plan.items()):
        nwp_init_datetime_str = nwp_init_dt.strftime("%Y-%m-%dT%HZ")
        dst_path = dst_root / nwp_init_datetime_str
        batch_info_str = f"Batch [{i + 1}/{n_batches}]"
        log.info(
            "%s starting: Asking rclone to copy %d file(s) totalling %s to %s (if they don't already exist)...",
            batch_info_str,
            len(files_to_be_copied),
            format_bytes(_sum_bytes(files_to_be_copied)),
            dst_path / nwp_var,
        )
        batch_summary = _copy_batch(
            ftp_host=ftp_host,
            ftp_path=ftp_path,
            dst_path=dst_path,
            files_to_be_copied=files_to_be_copied,
            transfers=transfers,
            env_vars=env_vars,
        )
        log.info("%s complete: %s", batch_info_str, batch_summary)
        total_summary += batch_summary

    log.info("Transfer from %s to %s complete: %s", ftp_path, dst_root, total_summary)
    return total_summary


def _copy_batch(
    ftp_host: str,
    ftp_path: PurePosixPath,
    dst_path: PurePosixPath,
    files_to_be_copied: list[_PathAndSize],
    transfers: int,
    env_vars: dict[str, Any] | None = None,
) -> TransferSummary:
    """Executes a single rclone copy batch for a specific destination.

    rclone will only transfer files that are missing from dst_path, or are present in dst_path but
    have a different size (e.g. a partial copy from a previous transfer that crashed).

    Args:
        ftp_host: The FTP host (e.g. 'opendata.dwd.de').
        ftp_path: The root source path on the FTP server (including NWP init hour).
        dst_path: The specific destination directory (including NWP init timestamp).
        files_to_be_copied: List of file path and sizes relative to ftp_path to be copied.
        transfers: Number of parallel transfers to use for this batch.
        env_vars: Additional environment variables to give to `rclone`.
    """
    # Modern Linux platforms often install `rclone` as a sandboxed snap, which does not have access
    # to `/tmp`, so we store the temporary file in the current working directory.
    with NamedTemporaryFile(mode="w", dir=".", prefix=".rclone_files_") as list_file:
        # paths in `files_to_be_copied` are relative to `ftp_path`.
        list_file.write("\n".join([p.path.as_posix() for p in files_to_be_copied]))
        list_file.flush()

        cmd = [
            "rclone",
            "copy",
            "--verbose",
            f":ftp:{ftp_path}",
            str(dst_path),
            "--files-from-raw=" + list_file.name,
            f"--transfers={transfers}",
            "--ignore-checksum",  # DWD's FTP server does not support hashing.
            "--update",  # Skip files that are newer on the destination.
            "--fast-list",  # Use less API calls to S3, in exchange for using more RAM.
            *_get_rclone_ftp_args(ftp_host),
            *_get_common_rclone_args(),
        ]

        try:
            _, log_entries = _run_rclone(
                cmd, timeout=timedelta(minutes=30), env_vars=env_vars
            )
        except subprocess.CalledProcessError:
            log.exception("Failed to copy batch to %s", dst_path)
            raise

    return TransferSummary.from_rclone_stats(log_entries)


def _get_rclone_ftp_args(ftp_host: str) -> list[str]:
    return [
        "--ftp-host=" + ftp_host,
        "--ftp-user=anonymous",
        # rclone requires passwords to be obscured by encrypting & encoding them in base64.
        # The base64 string below was created with the command `rclone obscure guest`.
        "--ftp-pass=JUznDm8DV5bQBCnXNVtpK3dN1qHB",
    ]


def _get_common_rclone_args() -> list[str]:
    return [
        "--use-json-log",
        "--config=",  # There is no config file because we pass everything as command-line args.
    ]


def _run_rclone(
    cmd: list[str],
    timeout: timedelta,
    env_vars: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Runs a command with logging and safety measures and returns (stdout, log_entries)."""
    cmd_str = " ".join(cmd)
    log.debug("Running: `%s`", cmd_str)

    full_env = os.environ.copy()
    if env_vars:
        full_env.update(env_vars)

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,  # Open stdout and stderr pipes in text mode (not bytes).
            check=True,  # Raise CalledProcessError if returncode != 0.
            timeout=round(timeout.total_seconds()),
            preexec_fn=_set_death_signal,
            env=full_env,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        stderr_str = (
            e.stderr
            if isinstance(e.stderr, str)
            else (e.stderr.decode() if e.stderr is not None else "")
        )
        log.exception(
            "Exception when running command `%s`. stdout='%s'",
            cmd_str,
            stderr_str,
        )
        raise

    try:
        log_entries = parse_and_log_rclone_json(result.stderr)
    except:
        log.exception(
            "Failed to parse JSON output from rclone. rclone's stderr='%s'. Command=`%s`",
            result.stderr,
            cmd_str,
        )
        raise
    return result.stdout, log_entries


def _set_death_signal() -> None:
    """Linux-specific: Ensure the child process dies if the parent dies."""
    pr_set_pdeathsig: Final[int] = 1  # For Linux prctl (PRocess ConTroL)
    libc = ctypes.CDLL("libc.so.6")
    # Send SIGTERM to the child if the parent terminates.
    libc.prctl(pr_set_pdeathsig, signal.SIGTERM)
