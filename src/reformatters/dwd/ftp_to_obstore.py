import ctypes
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import PurePosixPath
from typing import ReadOnly, TypedDict

from reformatters.common.logging import get_logger

log = get_logger(__name__)


def _log_rclone_stderr(stderr: str) -> None:
    for line in stderr.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue

        # rclone uses "Error", "Warning", "Info" in its output.
        line_upper = clean_line.upper()
        if "ERROR" in line_upper or "FAILED" in line_upper:
            log_level = logging.ERROR
        elif "WARNING" in line_upper:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        log.log(level=log_level, msg=f"Rclone: {clean_line}")


# Load the libc library to access prctl
libc = ctypes.CDLL("libc.so.6")


def _set_death_signal() -> None:
    """Send SIGTERM to the child process if the parent process (python) dies."""
    pr_set_pdeathsig = 1
    libc.prctl(pr_set_pdeathsig, signal.SIGTERM)


def _call_command_with_logging(cmd: list, timeout: int = 90) -> str:
    log.info("Command: %s", " ".join(cmd))

    # --------- Killing the subprocess when Python dies ---------------
    # The default behaviour for `subprocess.Popen` is that `process` will continue running as a
    # zombie process if python dies before `process` finishes. We don't want that! We can ensure
    # `process` is killed if python is killed. But there are two ways Python could be killed:
    #
    # 1. `kill -9 python` (SIGKILL) or `kill python` (SIGTERM) -> the OS instantly kills Python.
    #    Python doesn't have time to throw any exceptions. We set `preexec_fn` to handle this case.
    # 2. `Ctrl+C` -> the OS sends SIGINT to Python -> Python raises a KeyboardInterrupt -> We catch
    #     this and call `process.terminate()`. prctl DOES NOT fire here because Python is still "alive" handling the exception.

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # Rclone sends its progress and status messages to stderr.
        text=True,  # stdout and stderr will be opened in text mode (not bytes).
        bufsize=1,  # Enable line buffering for stdout and stderr.
        # preexec_fn will be called in the child process just before cmd is executed.
        # WARNING: preexec_fn is not supported in threaded code or in Python subinterpreters.
        preexec_fn=_set_death_signal,
    )
    try:
        # You might wonder why we're using `process.communicate` instead of `process.wait`.
        # `process.wait` deadlocks if the process returns lots of data in a PIPE (e.g. rclone
        # returning a long directory listing through stdout).
        stdout_str, stderr_str = process.communicate(timeout=timeout)
    except (KeyboardInterrupt, SystemExit):
        log.exception("Python is shutting down! Terminating rclone...")
        process.terminate()
        try:
            # Give `process` 5 seconds to wrap up:
            stdout_str, stderr_str = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()  # Force kill if it's stubborn
        else:
            _log_rclone_stderr(stderr_str)
            if stdout_str:
                log.info("Rclone stdout during shutdown: %s", stdout_str)
        raise

    _log_rclone_stderr(stderr_str)

    if process.returncode == 0:
        return stdout_str
    else:
        error_msg = f"rclone return code is {process.returncode}. stdout={stdout_str}"
        log.error(error_msg)
        raise RuntimeError(error_msg)


class _ListItem(TypedDict):
    """An item returned by rclone's lsjson command."""

    Path: ReadOnly[str]
    Name: ReadOnly[str]
    Size: ReadOnly[int]
    ModTime: ReadOnly[str]
    IsDir: ReadOnly[bool]


class FtpToObstore:
    def __init__(
        self, ftp_host: str, ftp_path: PurePosixPath, dst_path: PurePosixPath
    ) -> None:
        """
        Args:
            ftp_host: The FTP host, e.g. 'opendata.dwd.de'
            ftp_path: The source path on the FTP host, e.g. '/weather/nwp/icon-eu/grib/00'
            dst_path: The destination path.
        """
        self.ftp_host = ftp_host
        self.ftp_path = ftp_path
        self.dst_path = dst_path

    def ftp_list(self) -> list[_ListItem]:
        """
        Recursively list all files and directories below ftp_host/ftp_path.

        Returns a list of ListItem dictionaries. Note that the `Path` attribute in the returned dict
        does not include the `self.path`. For example, a returned `Path` might look like
        "aswdifd_s/icon-eu_europe_regular-lat-lon_single-level_2026011200_004_ASWDIFD_S.grib2.bz2"
        """
        log.info(f"Listing ftp://{self.ftp_host}{self.ftp_path} ...")
        cmd = [
            "rclone",
            "lsjson",  # List and return listing as JSON on stdout.
            "--recursive",
            "--fast-list",  # Optimizes listing for some remotes.
            "--no-mimetype",  # Don't read the mime type (can speed things up).
            "--no-modtime",  # Don't read the modification time (can speed things up).
            f":ftp:{self.ftp_path}",
            *self._ftp_host_and_user_and_pass,
            *self._common_params_for_rclone,
        ]

        stdout_str = _call_command_with_logging(cmd, timeout=90)

        try:
            return json.loads(stdout_str)
        except json.decoder.JSONDecodeError:
            log.exception(
                "Failed to decode stdout of rclone as json. stdout='%s'.",
                stdout_str,
            )
            raise

    @property
    def _ftp_host_and_user_and_pass(self) -> list[str]:
        return [
            "--ftp-host=" + self.ftp_host,
            "--ftp-user=anonymous",
            # rclone requires passwords to be obscured by encrypting & encoding them in base64.
            # The base64 string below was created with the command `rclone obscure guest`.
            "--ftp-pass=JUznDm8DV5bQBCnXNVtpK3dN1qHB",
        ]

    @property
    def _common_params_for_rclone(self) -> list[str]:
        return [
            "--config=",  # There is no rclone config file. We pass in all config as arguments.
        ]

    def generate_batches(
        self,
        file_list: list[_ListItem],
    ) -> dict[tuple[PurePosixPath, PurePosixPath], list[str]]:
        """
        Returns a dict which maps from (src_directory, dst_directory) to a list of filenames.

        For example:
        (
            PurePosixPath('/weather/nwp/icon-eu/grib/00/alb_rad'),
            PurePosixPath('/home/jack/data/ICON-EU/grib/rsync_and_python/2026-01-12T00Z/alb_rad')
        ): [
            "icon-eu_europe_regular-lat-lon_single-level_2026011200_000_ALB_RAD.grib2.bz2",
            "icon-eu_europe_regular-lat-lon_single-level_2026011200_001_ALB_RAD.grib2.bz2",
            ...
        ]
        """
        batches = defaultdict(list)
        # Regex to find the date in the filename (YYYYMMDDHH)
        date_regex = re.compile(r"_(\d{10})_")

        for file_info in file_list:
            if file_info["IsDir"]:
                continue

            file_path = PurePosixPath(file_info["Path"])

            # --- FILTERING ---
            if "pressure-level" in file_path.name:
                continue

            # --- PARSING ---
            match = date_regex.search(file_path.name)
            if not match:
                log.warning("Skipping (no date found): %s", file_path.name)
                continue

            raw_date = match.group(1)
            dt = datetime.strptime(raw_date, "%Y%m%d%H")
            date_dir = dt.strftime("%Y-%m-%dT%HZ")

            # Extract the NWP parameter name from path (e.g., 'alb_rad' from 'alb_rad/file')
            if len(file_path.parts) != 2:
                continue
            param_name = file_path.parts[0]  # e.g. 'alb_rad'

            src_dir_url = self.ftp_path / file_path.parent
            dest_dir_url = self.dst_path / date_dir / param_name

            batch_key = (src_dir_url, dest_dir_url)
            batches[batch_key].append(file_path.name)

        return batches

    def run_transfers(
        self, batches: dict[tuple[str, PurePosixPath], list[str]]
    ) -> None:
        """
        Iterates through batches, creates a temp file list, and runs rclone.
        """
        total_batches = len(batches)
        log.info("Processing %s batch groups...", total_batches)

        for i, ((src_url, dst_url), filenames) in enumerate(batches.items(), 1):
            if i < 40:
                continue
            elif i > 50:
                break

            log.info(
                f"[{i}/{total_batches}] Syncing {len(filenames)} files to {dst_url}..."
            )

            # Create a temporary file to hold the list of filenames for this batch
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
                tmp.write("\n".join(filenames))
                tmp_file_containing_filenames = tmp.name

            cmd = [
                "rclone",
                "copy",
                ":ftp:" + str(src_url),
                str(dst_url),
                "--files-from=" + tmp_file_containing_filenames,
                "--transfers=10",  # Increase parallelism.
                "--no-check-certificate",
                "--no-traverse",
                *self._ftp_host_and_user_and_pass,
                *self._common_params_for_rclone,
            ]

            try:
                stdout_str = _call_command_with_logging(cmd, timeout=60 * 5)
            except subprocess.CalledProcessError as e:
                log.exception(f"Error syncing batch {src_url}: {e}")
            else:
                if stdout_str:
                    log.info("Rclone stdout: %s", stdout_str)
            finally:
                os.remove(tmp_file_containing_filenames)
