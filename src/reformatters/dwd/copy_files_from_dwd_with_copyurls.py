import os
import re
import subprocess
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from subprocess import PIPE, CalledProcessError
from typing import IO, Any, Final

from reformatters.common.logging import get_logger

log = get_logger(__name__)


def copy_gribs_from_dwd_https(
    src_root_path: PurePosixPath,
    dst_root_path: PurePosixPath,
    src_host: str = "https://opendata.dwd.de",
    transfer_parallelism: int = 64,
    checkers: int = 16,
    stats_logging_freq: str = "1m",
    env_vars: dict[str, Any] | None = None,
) -> None:
    """
    Args:
        src_root_path: The absolute path on src_host for one NWP run.
            Must start with a forwards slash, e.g. "/weather/nwp/icon-eu/grib/00/"
        dst_root_path: The destination path, e.g. "/data/ICON-EU/" or ":s3:bucket/path".
            Must be in the format that rclone expects.
        src_host: The HTTP or HTTPS URL, e.g. "https://opendata.dwd.de".
            Should not include a trailing slash.
        transfer_parallelism: Number of concurrent workers during the copy operation.
            Each worker fetches a file from src_host, copies it to the destination, and waits for
            the destination to acknowledge completion before fetching another file from the source.
            When fetching from HTTPS and writing to object storage, this could be set arbitrarily
            high, although setting it too high (>256?) might be detrimental to performance.
        stats_logging_freq: The period between each stats log. e.g. "1m" to log stats every minute.
            See https://rclone.org/docs/#stats-duration
        env_vars: Additional environment variables to give to `rclone`. For example:
            {
                "RCLONE_S3_PROVIDER": "AWS",
                "RCLONE_S3_ACCESS_KEY_ID": "key",
                "RCLONE_S3_SECRET_ACCESS_KEY": "secret",
                "RCLONE_S3_REGION": "us-west-2",
            }
    """
    # Check inputs:
    if src_host[-1] == "/":
        log.info("Stripping trailing slash from src_host %s", src_host)
        src_host = src_host[:-1]
    if not src_root_path.is_absolute():
        raise ValueError(
            "src_root_path '%s' must start with a forward slash.", src_root_path
        )

    # Set full_env variables:
    if env_vars:
        full_env = os.environ.copy()
        full_env.update(env_vars)
    else:
        full_env = None

    src_paths_starting_with_nwp_var = list_grib_files_on_dwd_https(
        http_url=src_host,
        path=src_root_path,
        checkers=checkers,
        env_vars=full_env,
    )

    files_already_on_dst = list_files_on_dst_for_all_nwp_runs_available_from_dwd(
        src_paths_starting_with_nwp_var=src_paths_starting_with_nwp_var,
        src_root_path_ending_with_init_hour=src_root_path,
        dst_root_path_without_init_dt=dst_root_path,
        checkers=checkers,
        env_vars=full_env,
    )

    csv_of_files_to_transfer = compute_which_files_still_need_to_be_transferred(
        src_paths_starting_with_nwp_var=src_paths_starting_with_nwp_var,
        files_already_on_dst=files_already_on_dst,
        src_host_and_root_path=f"{src_host}{src_root_path}",
    )

    run_rclone_copyurl(
        "\n".join(csv_of_files_to_transfer),
        dst_root_path=dst_root_path,
        transfer_parallelism=transfer_parallelism,
        checkers=checkers,
        env_vars=full_env,
        stats_logging_freq=stats_logging_freq,
    )


def list_grib_files_on_dwd_https(
    http_url: str,
    path: str | PurePosixPath,
    checkers: int,
    env_vars: dict[str, Any] | None = None,
) -> list[PurePosixPath]:
    rclone_args = (
        f"--http-url={http_url}",
        "--min-age=1m",  # Ignore files that are so young they might be incomplete.
        # The ordering of these filters matters:
        "--filter=- *pressure-level*",
        "--filter=+ *.grib2.bz2",
        "--filter=- *",
    )
    return list_files(
        path=f":http:{path}",
        checkers=checkers,
        rclone_args=rclone_args,
        env_vars=env_vars,
    )


def list_files(
    path: str,
    checkers: int,
    rclone_args: Sequence[str] = (),
    env_vars: dict[str, Any] | None = None,
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
        env_vars: Additional environment variables to give to `rclone`.

    Returns:
        paths: A sorted list of all the files found in `path`. Returns an empty list if the
        directory does not exist.
    """
    log.info("Listing files on '%s'...", path)
    cmd = (
        "rclone",
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
            env=env_vars,
            timeout=120,
        )
    except CalledProcessError as e:
        if (
            e.returncode == 3
            and isinstance(e.stderr, str)
            and "directory not found" in e.stderr.lower()
        ):
            log.info("Directory not found: '%s'", path)
            return []
        else:
            log_error_from_called_process_error(e)
            raise
    else:
        if result.stderr:
            log.info("rclone stderr: %s", result.stderr)
        paths = sorted(PurePosixPath(p) for p in result.stdout.splitlines())
        log.info(f"Found {len(paths):,d} files on '{path}'.")
        return paths


def convert_src_path_to_dst_path(
    src_path_starting_with_nwp_var: PurePosixPath,
) -> PurePosixPath:
    """
    Args:
        src_path_starting_with_nwp_var: Should have exactly 2 parts: NWP_var/filename.grib2.bz2.
    Returns:
        dst_path: In the form NWP_init_datetime/NWP_var/filename.grib2.bz2.
    """
    grib_filename = src_path_starting_with_nwp_var.name
    nwp_init_datetime = extract_nwp_init_datetime_from_grib_filename(grib_filename)
    nwp_init_datetime_str = format_datetime_for_dst_path(nwp_init_datetime)
    nwp_variable_name = extract_nwp_var_from_src_path(src_path_starting_with_nwp_var)
    dst_path = PurePosixPath(nwp_init_datetime_str) / nwp_variable_name / grib_filename
    return dst_path


def extract_nwp_var_from_src_path(src_path: PurePosixPath) -> str:
    """Check number of parts in the path & extract NWP variable name.

    Args:
        src_path: A path on the DWD HTTPs server in the form:
            alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026013003_000_ALB_RAD.grib2.bz2
    """
    n_expected_parts: Final[int] = 2
    if len(src_path.parts) != n_expected_parts:
        raise RuntimeError(
            "Expected %d parts in the DWD path, found %d parts. Path: %s",
            n_expected_parts,
            len(src_path.parts),
            src_path,
        )
    return src_path.parts[0]


class DateExtractionError(Exception):
    pass


def extract_nwp_init_datetime_from_grib_filename(
    grib_filename: str,
) -> datetime:
    dwd_nwp_init_date_regex: Final[re.Pattern[str]] = re.compile(r"_(\d{10})_")
    nwp_init_date_matches = dwd_nwp_init_date_regex.findall(grib_filename)
    if len(nwp_init_date_matches) == 0:
        raise DateExtractionError("No date found in file: %s", grib_filename)
    elif len(nwp_init_date_matches) > 1:
        raise DateExtractionError(
            "Expected exactly one 10-digit number in the filename (the NWP init date"
            " represented as YYYYMMDDHH), but instead found %d 10-digit numbers in path %s",
            len(nwp_init_date_matches),
            grib_filename,
        )
    nwp_init_date_str = nwp_init_date_matches[0]
    return datetime.strptime(nwp_init_date_str, "%Y%m%d%H")


def format_datetime_for_dst_path(nwp_init_datetime: datetime) -> str:
    return nwp_init_datetime.strftime("%Y-%m-%dT%H")


def list_files_on_dst_for_all_nwp_runs_available_from_dwd(
    src_paths_starting_with_nwp_var: Sequence[PurePosixPath],
    src_root_path_ending_with_init_hour: PurePosixPath,
    dst_root_path_without_init_dt: PurePosixPath,
    checkers: int,
    env_vars: dict[str, Any] | None = None,
) -> set[PurePosixPath]:
    """The returned paths include (and start with) the NWP init datetime."""
    # Find unique NWP runs available from DWD. Usually, a DWD path like
    # `/weather/nwp/icon-eu/grib/00/` will only contain files for a single NWP run (today's midnight
    # run). But, if the time now is between 2 hours and 4 hours after the init time, then DWD will
    # be in the process of overwriting the files for yesterday's midnight run with today's midnight
    # run, and the 00/ directory will contain files from two NWP runs.
    unique_nwp_init_datetimes: set[datetime] = {
        extract_nwp_init_datetime_from_grib_filename(src_path.name)
        for src_path in src_paths_starting_with_nwp_var
    }
    log.info(
        f"Found {len(unique_nwp_init_datetimes)} unique NWP init datetime(s)"
        f" in '{src_root_path_ending_with_init_hour}': {unique_nwp_init_datetimes}"
    )

    # Get a set of all the files in the destination:
    # The paths in this set start with and *include* the NWP init datetime part of the path.
    existing_dst_paths_starting_with_init_dt: set[PurePosixPath] = set()
    for nwp_init_dt in unique_nwp_init_datetimes:
        nwp_init_dt_str = format_datetime_for_dst_path(nwp_init_dt)
        dst_paths_starting_with_nwp_var = list_files(
            str(dst_root_path_without_init_dt / nwp_init_dt_str),
            env_vars=env_vars,
            checkers=checkers,
        )
        existing_dst_paths_starting_with_init_dt.update(
            nwp_init_dt_str / dst_path for dst_path in dst_paths_starting_with_nwp_var
        )

    return existing_dst_paths_starting_with_init_dt


def compute_which_files_still_need_to_be_transferred(
    src_paths_starting_with_nwp_var: Sequence[PurePosixPath],
    files_already_on_dst: set[PurePosixPath],
    src_host_and_root_path: str,
) -> list[str]:
    """Returns list of strings, each of which is a row of a CSV with two columns:

    1. The full source path, e.g. `https://opendata.dwd.de/.../filename.grib2.bz2`.
    2. The destination path, from the NWP init datetime onwards.

    This is the format required by `rclone copyurls`.
    """
    csv_of_files_to_transfer: list[str] = []  # Each list item is one line of the CSV.
    for src_path in src_paths_starting_with_nwp_var:
        dst_path = convert_src_path_to_dst_path(src_path)
        if dst_path not in files_already_on_dst:
            full_src_path = src_host_and_root_path / src_path
            dst_path = convert_src_path_to_dst_path(src_path)
            csv_of_files_to_transfer.append(f"{full_src_path},{dst_path}")
    log.info(f"Planning to transfer {len(csv_of_files_to_transfer):,d} files.")
    return csv_of_files_to_transfer


def run_rclone_copyurl(
    csv_of_files_to_transfer: str,
    dst_root_path: PurePosixPath,
    transfer_parallelism: int,
    checkers: int,
    stats_logging_freq: str,  # e.g. "1m" for every 1 minute.
    env_vars: dict[str, Any] | None = None,
) -> None:
    csv_file = Path("copyurls.csv")
    csv_file.write_text(csv_of_files_to_transfer)
    cmd = (
        "rclone",
        "copyurl",  # https://rclone.org/commands/rclone_copyurl
        "--urls",
        str(csv_file),
        str(dst_root_path),
        # Performance:
        "--fast-list",
        f"--transfers={transfer_parallelism:d}",
        f"--checkers={checkers:d}",
        # Logging:
        f"--stats={stats_logging_freq}",
        "--stats-log-level=ERROR",  # Output stats to stderr.
        "--quiet",  # Only output logs at error level.
        "--stats-one-line",  # Output stats as a single line.
    )
    run_command_with_concurrent_logging(cmd, env_vars=env_vars)
    csv_file.unlink()


def run_command_with_concurrent_logging(
    cmd: Sequence[str],
    env_vars: dict[str, Any] | None = None,
) -> int:
    cmd_str = " ".join(cmd)
    log.info("Running command: %s", cmd_str)

    try:
        process = subprocess.Popen(  # noqa: S603
            cmd, text=True, stdout=PIPE, stderr=PIPE, bufsize=1, env=env_vars
        )

        # Create threads to read stdout and stderr simultaneously
        t1 = threading.Thread(target=log_stdout, args=(process.stdout,))
        t2 = threading.Thread(target=log_stderr_stats, args=(process.stderr,))

        t1.start()
        t2.start()

        # Wait for threads to finish (which happens when process closes the pipes)
        t1.join()
        t2.join()

        return_code = process.wait()
    except CalledProcessError as e:
        log_error_from_called_process_error(e)
        raise
    except KeyboardInterrupt:
        # Avoid having a zombie rclone process if user kills Python with Ctrl-C
        log.warning("Received KeyboardInterrupt... terminating subprocess...")
        process.terminate()
        raise
    else:
        log.info("return code = %d after running command: '%s'", return_code, cmd_str)
        return return_code


def log_error_from_called_process_error(e: CalledProcessError) -> None:
    log.exception(
        "stderr: %s; stdout: %s",
        convert_called_process_error_output_to_str(e.stderr),
        convert_called_process_error_output_to_str(e.stdout),
    )


def convert_called_process_error_output_to_str(output: None | str | bytes) -> str:
    if output is None:
        return ""
    elif isinstance(output, str):
        return output
    else:
        return output.decode()


def log_stdout(pipe: IO[str]) -> None:
    """Reads a pipe line-by-line and logs it."""
    with pipe:
        for line in pipe:
            log.info(f"stdout: {line.strip()}")


def log_stderr_stats(pipe: IO[str]) -> None:
    with pipe:
        for line in pipe:
            try:
                tidy_line = tidy_stats(line)
            except Exception:  # noqa: BLE001
                # An exception here just means the line wasn't a stats line,
                # so let's log it and move on. No biggie.
                log.info("stderr: '%s'", line)
            else:
                log.info(f"Rclone stats: {tidy_line}")


def tidy_stats(line: str) -> str:
    """Remove meaningless (and hence confusing) numbers from rclone stats!

    Example raw stats output from rclone copyurl:

        2026/01/31 16:15:41 ERROR :    16.342 MiB / 18.818 MiB, 87%, 0 B/s, ETA -
                            ^^^^^                 ^^^^^^^^^^^^  ^^^         ^^^^^
    Issues to fix:    Stats aren't an error!      And these numbers means nothing!
    """
    # Split by the first colon to ignore the timestamp and 'ERROR'
    split_on: Final[str] = "ERROR :"
    if split_on not in line:
        raise ValueError("Expected a colon in rclone stats line: '%s'", line)
    line = line.split(split_on, 1)[1]

    # Split the remaining data by comma
    # parts[0] = "16.342 MiB / 18.818 MiB" (Size info)
    # parts[1] = " 87%" (Percentage)
    # parts[2] = " 0 B/s" (Speed)
    # parts[3] = " ETA -"
    parts = line.split(",")
    n_expected_parts: Final[int] = 4
    if len(parts) != n_expected_parts:
        raise ValueError(
            "Expected %d comma-separated values in rclone stats line. Line: '%s'",
            n_expected_parts,
            line,
        )

    transferred_bytes = parts[0].split("/")[0].strip()
    speed = parts[2].strip()
    return f"Transferred so far: {transferred_bytes}. Recent throughput: {speed}"


if __name__ == "__main__":
    copy_gribs_from_dwd_https(
        src_root_path=PurePosixPath("/weather/nwp/icon-eu/grib/03/"),
        dst_root_path=PurePosixPath("/home/jack/data/ICON-EU/grib/rclone_copyurls/"),
    )
