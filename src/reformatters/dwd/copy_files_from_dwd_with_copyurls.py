import re
import subprocess
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from subprocess import PIPE, CompletedProcess
from typing import IO, Final

from reformatters.common.logging import get_logger

log = get_logger(__name__)


def list_files(
    path: str, checkers: int | None = None, rclone_args: Sequence[str] = ()
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
    """
    log.info("Listing files on %s", path)
    checkers = checkers or 32
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
    result = run_command(cmd)
    paths = sorted(PurePosixPath(p) for p in result.stdout.splitlines())
    log.info(f"Found {len(paths):,d} files on '{path}'.")
    return paths


def list_grib_files_on_dwd_https(
    path: str, checkers: int | None = None
) -> list[PurePosixPath]:
    rclone_args = (
        "--min-age=1m",  # Ignore files that are so young they might be incomplete.
        # The ordering of these filters matters:
        "--filter=- *pressure-level*",
        "--filter=+ *.grib2.bz2",
        "--filter=- *",
    )
    return list_files(path=path, checkers=checkers, rclone_args=rclone_args)


def extract_nwp_var_from_dwd_path(dwd_path: PurePosixPath) -> str:
    """Check number of parts in the path & extract NWP variable name.

    Args:
        dwd_path: A path on the DWD HTTPs server in the form:
            alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026013003_000_ALB_RAD.grib2.bz2
    """
    n_expected_parts: Final[int] = 2
    if len(dwd_path.parts) != n_expected_parts:
        raise RuntimeError(
            "Expected %d parts in the DWD path, found %d parts. Path: %s",
            n_expected_parts,
            len(dwd_path.parts),
            dwd_path,
        )
    return dwd_path.parts[0]


class DateExtractionError(Exception):
    pass


def extract_nwp_init_datetime_from_dwd_path(
    dwd_path_without_root: PurePosixPath,
) -> datetime:
    dwd_nwp_init_date_regex: Final[re.Pattern[str]] = re.compile(r"_(\d{10})_")
    nwp_init_date_match = dwd_nwp_init_date_regex.findall(dwd_path_without_root.name)
    if len(nwp_init_date_match) == 0:
        raise DateExtractionError("No date found in file: %s", dwd_path_without_root)
    elif len(nwp_init_date_match) > 1:
        raise DateExtractionError(
            "Expected exactly one 10-digit number in the filename (the NWP init date"
            " represented as YYYYMMDDHH), but instead found %d 10-digit numbers in path %s",
            len(nwp_init_date_match),
            dwd_path_without_root,
        )
    nwp_init_date_str = nwp_init_date_match[0]
    return datetime.strptime(nwp_init_date_str, "%Y%m%d%H")


def format_datetime_for_use_in_dst_path(nwp_init_datetime: datetime) -> str:
    return nwp_init_datetime.strftime("%Y-%m-%dT%H")


def convert_dwd_path_to_dst_path(dwd_path: PurePosixPath) -> PurePosixPath:
    """
    Args:
        dwd_path: Should have exactly 2 parts: NWP_var/filename.grib2.bz2.
    Returns:
        dst_path: In the form NWP_init_datetime/NWP_var/filename.grib2.bz2.
    """
    nwp_init_datetime = extract_nwp_init_datetime_from_dwd_path(dwd_path)
    nwp_init_datetime_str = format_datetime_for_use_in_dst_path(nwp_init_datetime)
    nwp_var = extract_nwp_var_from_dwd_path(dwd_path)
    dst_path = PurePosixPath(nwp_init_datetime_str) / nwp_var / dwd_path.name
    return dst_path


def run_command(cmd: Sequence[str], log_stdout: bool = False) -> CompletedProcess[str]:
    log.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if result.stderr:
        log.info("rclone stderr: %s", result.stderr)
    if log_stdout and result.stdout:
        log.info("rclone stdout: %s", result.stdout)
    return result


def run_command_with_concurrent_logging(cmd: Sequence[str]) -> int:
    log.info("Running command: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, text=True, stdout=PIPE, stderr=PIPE, bufsize=1)

    # Create threads to read stdout and stderr simultaneously
    t1 = threading.Thread(target=stream_reader, args=(process.stdout, "stdout"))
    t2 = threading.Thread(target=stream_reader, args=(process.stderr, "stderr"))

    t1.start()
    t2.start()

    # Wait for threads to finish (which happens when process closes the pipes)
    t1.join()
    t2.join()

    return_code = process.wait()
    log.info("return code = %s", return_code)
    return return_code


def stream_reader(pipe: IO[str], prefix: str) -> None:
    """Reads a pipe line-by-line and logs it."""
    with pipe:
        for line in pipe:
            log.info(f"{prefix}: {line.strip()}")


def run_rclone_copyurls(csv_of_files_to_transfer: str) -> None:
    csv_file = Path("copyurls.csv")
    csv_file.write_text(csv_of_files_to_transfer)

    cmd = [
        "rclone",
        "copyurl",
        "--urls",
        "copyurls.csv",
        "/home/jack/data/ICON-EU/grib/rclone_copyurls/",  # TODO(Jack): Replace this with dst_root_path
        # Performance:
        "--fast-list",
        "--transfers=16",  # TODO(Jack): transfers and checkers should be configurable.
        "--checkers=16",
        # Logging:
        "--stats=2s",  # Output statistics every 2 seconds.
        "--use-json-log",  # Output all logs in JSON.
        # "--progress",
        # "--quiet",
        # "--stats-one-line",
        # "--log-level=ERROR",
        # "--stats-log-level=NOTICE",
    ]
    run_command_with_concurrent_logging(cmd)

    # csv_file.unlink()  # TODO(Jack): uncomment this after testing!!!


def list_files_on_dst_for_nwp_runs_available_from_dwd(
    dwd_paths_without_root: Sequence[PurePosixPath],
    dwd_root_path: PurePosixPath,
    dst_root_path: PurePosixPath,
) -> list[PurePosixPath]:
    """The returned paths include (and start with) the NWP init datetime."""
    # Find unique NWP runs available from DWD. Usually, a DWD path like
    # `/weather/nwp/icon-eu/grib/00/` will only contain files for a single NWP run (today's midnight
    # run). But, if the time now is between 2 hours and 4 hours after the init time, then DWD will
    # be in the process of overwriting the files for yesterday's midnight run with today's midnight
    # run, and the 00/ directory will contain files from two NWP runs.
    unique_nwp_init_datetimes = {
        extract_nwp_init_datetime_from_dwd_path(dwd_path_without_root)
        for dwd_path_without_root in dwd_paths_without_root
    }
    log.info(
        f"Found {len(unique_nwp_init_datetimes)} unique NWP init datetime(s) in {dwd_root_path}: {unique_nwp_init_datetimes}"
    )

    # Get a list of all the files in the destination:
    # The paths in this list start with and *include* the NWP init datetime part of the path.
    existing_dst_paths_starting_with_init_dt: list[PurePosixPath] = []
    for nwp_init_dt in unique_nwp_init_datetimes:
        nwp_init_dt_str = format_datetime_for_use_in_dst_path(nwp_init_dt)
        dst_paths_starting_with_nwp_var = list_files(
            str(dst_root_path / nwp_init_dt_str)
        )
        existing_dst_paths_starting_with_init_dt.extend(
            nwp_init_dt_str / dst_path for dst_path in dst_paths_starting_with_nwp_var
        )

    return existing_dst_paths_starting_with_init_dt


def main() -> None:
    dwd_root_path = PurePosixPath("/weather/nwp/icon-eu/grib/03/")
    dst_root_path = PurePosixPath("/home/jack/data/ICON-EU/grib/rclone_copyurls/")

    # These dwd_paths do not include the dwd_root_path.
    dwd_paths_without_root = list_grib_files_on_dwd_https(
        path=f"dwd-http:{dwd_root_path}",  # rclone_remote:path
    )

    files_already_on_dst = list_files_on_dst_for_nwp_runs_available_from_dwd(
        dwd_paths_without_root=dwd_paths_without_root,
        dwd_root_path=dwd_root_path,
        dst_root_path=dst_root_path,
    )

    # Prepare a CSV with two columns:
    # 1. The source path, from `https://opendata.dwd.de` onwards.
    # 2. The destination path, from the NWP init datetime onwards.
    # This is the format required by `rclone copyurls`.
    csv_of_files_to_transfer: list[str] = []  # Each list item is one line of the CSV.
    for dwd_path in dwd_paths_without_root:
        dst_path = convert_dwd_path_to_dst_path(dwd_path)
        if dst_path not in files_already_on_dst:
            src_path = f"https://opendata.dwd.de{dwd_root_path / dwd_path}"
            dst_path = convert_dwd_path_to_dst_path(dwd_path)
            csv_of_files_to_transfer.append(f"{src_path},{dst_path}")
    log.info("Planning to transfer %d files.", len(csv_of_files_to_transfer))

    run_rclone_copyurls("\n".join(csv_of_files_to_transfer))

    # TODO(Jack): Stream stats? Maybe use json stats again? With an update every minute?


if __name__ == "__main__":
    main()
