import ctypes
import json
import logging
import re
import signal
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import PurePosixPath
from typing import Final, NamedTuple, ReadOnly, TypedDict

from reformatters.common.logging import get_logger

log = get_logger(__name__)


class _FtpListItem(TypedDict):
    """An item returned by rclone's lsjson command."""

    Path: ReadOnly[str]
    Name: ReadOnly[str]
    Size: ReadOnly[int]
    ModTime: ReadOnly[str]
    IsDir: ReadOnly[bool]


class _SrcAndDstPath(NamedTuple):
    src: PurePosixPath
    dst: PurePosixPath


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

    def ftp_list(self) -> list[_FtpListItem]:
        """
        Recursively list all files and directories below ftp_host/ftp_path.

        Returns a list of ListItem dictionaries. Note that the `Path` attribute in the returned dict
        does not include `self.path`. For example, a returned `Path` might look like
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
            *self._ftp_host_and_credentials,
            *self._common_params_for_rclone,
        ]

        stdout_str = _call_command_with_logging(cmd, timeout=90)

        try:
            decoded_json: list[_FtpListItem] = json.loads(stdout_str)
        except json.decoder.JSONDecodeError:
            log.exception(
                "Failed to decode stdout of rclone as json. stdout='%s'.",
                stdout_str,
            )
            raise
        else:
            return decoded_json

    @property
    def _ftp_host_and_credentials(self) -> list[str]:
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

    def compute_directories_to_copy(
        self,
        file_list: list[_FtpListItem],
    ) -> dict[_SrcAndDstPath, list[str]]:
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
        dirs_and_files_to_copy: dict[_SrcAndDstPath, list[str]] = defaultdict(list)
        n_skipped_files_per_src_dir: dict[PurePosixPath, int] = defaultdict(int)

        # Regex to find the date in the ICON-EU grib filename (YYYYMMDDHH)
        date_regex = re.compile(r"_(\d{10})_")

        n_expected_path_parts: Final[int] = 2

        for file_info in file_list:
            if file_info["IsDir"]:
                continue

            # `file_path` should be of the form:
            # alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011200_001_ALB_RAD.grib2.bz2
            file_path = PurePosixPath(file_info["Path"])
            src_dir = self.ftp_path / file_path.parent

            # --- FILTERING ---
            if "pressure-level" in file_path.name:
                n_skipped_files_per_src_dir[src_dir] += 1
                continue

            if len(file_path.parts) != n_expected_path_parts:
                n_skipped_files_per_src_dir[src_dir] += 1
                log.warning(
                    "Expected %d parts in the path, not %d. Skipping FTP filename: %s",
                    n_expected_path_parts,
                    len(file_path.parts),
                    file_path,
                )
                continue

            # --- PARSING ---
            match = date_regex.search(file_path.name)
            if not match:
                log.warning("Skipping file (no date found): %s", file_path.name)
                continue

            src_nwp_init_datetime_str = match.group(1)
            nwp_init_datetime = datetime.strptime(src_nwp_init_datetime_str, "%Y%m%d%H")
            dst_nwp_init_datetime_str = nwp_init_datetime.strftime("%Y-%m-%dT%HZ")

            nwp_param = file_path.parts[0]

            dst_dir = self.dst_path / dst_nwp_init_datetime_str / nwp_param

            src_and_dst = _SrcAndDstPath(src_dir, dst_dir)
            dirs_and_files_to_copy[src_and_dst].append(file_path.name)

        log.info(
            "Number of files skipped per directory: %s", n_skipped_files_per_src_dir
        )

        return dirs_and_files_to_copy

    def copy_directories(
        self, dirs_and_files_to_copy: dict[_SrcAndDstPath, list[str]]
    ) -> None:
        n_dirs = len(dirs_and_files_to_copy)
        log.info("Copying %d directories from ftp://%s", n_dirs, self.ftp_host)

        for i, ((src_url, dst_url), filenames) in enumerate(
            dirs_and_files_to_copy.items(), 1
        ):
            info_str = f"copying {len(filenames)} files from ftp://{self.ftp_host}{src_url} to {dst_url}"
            log.info("Directory [%d/%d]: %s...", i, n_dirs, info_str)

            # Create a temporary file to hold the list of filenames for this batch
            with tempfile.NamedTemporaryFile(mode="w+") as temp_file_of_filenames:
                temp_file_of_filenames.write("\n".join(filenames))

                cmd = [
                    "rclone",
                    "copy",
                    ":ftp:" + str(src_url),
                    str(dst_url),
                    "--files-from=" + temp_file_of_filenames.name,
                    "--transfers=10",  # Increase parallelism.
                    "--no-check-certificate",
                    "--no-traverse",
                    *self._ftp_host_and_credentials,
                    *self._common_params_for_rclone,
                ]

                try:
                    stdout_str = _call_command_with_logging(cmd, timeout=60 * 5)
                except subprocess.CalledProcessError as e:
                    log.exception("Error %s: %e", info_str, e)
                else:
                    if stdout_str:
                        log.debug("rclone stdout: %s", stdout_str)


def _call_command_with_logging(cmd: list[str], timeout: int = 90) -> str:
    log.info("Command: %s", " ".join(cmd))

    # --------- Killing the subprocess when Python dies ---------------
    # The default behaviour for `subprocess.Popen` is that `process` will continue running as a
    # zombie process if python dies before `process` finishes. We don't want that! We can ensure
    # `process` is killed if python is killed. But there are two ways Python could be killed:
    #
    # 1. `kill -9 python` (SIGKILL) or `kill python` (SIGTERM) -> the OS instantly kills Python.
    #    Python doesn't have time to throw any exceptions. We set `preexec_fn` to handle this case.
    # 2. `Ctrl+C` -> the OS sends SIGINT to Python -> Python raises a KeyboardInterrupt -> We catch
    #    this and call `process.terminate()`. prctl DOES NOT fire here because Python is still
    #    "alive" whilst handling the exception.

    process_name = cmd[0]
    process = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # rclone sends its progress and status messages to stderr.
        text=True,  # Open stdout and stderr text mode (not bytes).
        bufsize=1,  # Enable line buffering for stdout and stderr.
        # preexec_fn will be called in the child process just before cmd is executed.
        # WARNING: preexec_fn is not supported in threaded code or in Python subinterpreters.
        preexec_fn=_set_death_signal,  # noqa: PLW1509
    )
    try:
        # You might wonder why we're using `process.communicate` instead of `process.wait`.
        # `process.wait` deadlocks if the process returns lots of data in a pipe. In our case,
        # `process.wait` deadlocks when `rclone` returns a long directory listing through stdout.
        stdout_str, stderr_str = process.communicate(timeout=timeout)
    except (KeyboardInterrupt, SystemExit):
        _terminate_child_process(process, process_name)
        raise

    _log_rclone_stderr(stderr_str)

    if process.returncode == 0:
        return stdout_str
    else:
        raise subprocess.CalledProcessError(
            cmd=cmd, returncode=process.returncode, output=stdout_str, stderr=stderr_str
        )


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


def _set_death_signal() -> None:
    """Send SIGTERM to the child process if the parent process (python) dies."""
    libc = ctypes.CDLL("libc.so.6")  # Load the libc library to access prctl
    pr_set_pdeathsig = 1
    libc.prctl(pr_set_pdeathsig, signal.SIGTERM)


def _terminate_child_process(process: subprocess.Popen[str], process_name: str) -> None:
    log.exception("Terminating child process %s...", process_name)
    process.terminate()
    log.info(
        "Waiting 5 seconds for child %s process to terminate gracefully...",
        process_name,
    )
    try:
        stdout_str, stderr_str = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        log.warning(
            "Child process %s has not terminated gracefully. So we will forcefully kill it.",
            process_name,
        )
        process.kill()
    else:
        _log_rclone_stderr(stderr_str)
        if stdout_str:
            log.info("%s stdout during shutdown: %s", process_name, stdout_str)
