import re
import subprocess
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import PurePosixPath
from subprocess import PIPE, CompletedProcess
from typing import IO, Final

from reformatters.common.logging import get_logger

log = get_logger(__name__)


def list_files(
    path: str, checkers: int | None = None, rclone_args: Sequence[str] = ()
) -> list[PurePosixPath]:
    """The returned paths do not include the input `path`."""
    log.info("Listing files on %s", path)
    checkers = checkers or 32
    cmd = (
        "rclone",
        "lsf",
        path,
        "--fast-list",
        "--recursive",
        "--files-only",
        f"--checkers={checkers}",
        *rclone_args,
    )
    result = run_command(cmd)
    paths = sorted(PurePosixPath(p) for p in result.stdout.splitlines())
    log.info(f"Found {len(paths):,d} files on {path}")
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


def extract_nwp_init_datetime_from_dwd_path(dwd_path: PurePosixPath) -> datetime:
    dwd_nwp_init_date_regex: Final[re.Pattern[str]] = re.compile(r"_(\d{10})_")
    nwp_init_date_match = dwd_nwp_init_date_regex.findall(dwd_path.name)
    if len(nwp_init_date_match) == 0:
        raise RuntimeError("No date found in file: %s", dwd_path)
    elif len(nwp_init_date_match) > 1:
        raise ValueError(
            "Expected exactly one 10-digit number in the filename (the NWP init date"
            " represented as YYYYMMDDHH), but instead found %d 10-digit numbers in path %s",
            len(nwp_init_date_match),
            dwd_path,
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


def stream_reader(pipe: IO[str], info: str) -> None:
    """Reads a pipe line-by-line and logs it using the provided function."""
    with pipe:
        for line in pipe:
            log.info(f"{info}: {line.strip()}")


def run_command(cmd: Sequence[str], log_stdout: bool = False) -> CompletedProcess[str]:
    log.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if result.stderr:
        log.info("rclone stderr: %s", result.stderr)
    if log_stdout and result.stdout:
        log.info("rclone stdout: %s", result.stdout)
    return result


def run_command_with_concurrent_logging(cmd: Sequence[str]) -> None:
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


def run_rclone_copyurls(csv_of_files_to_transfer: str) -> None:
    # csv_file = Path("copyurls.csv")
    # csv_file.write_text(csv_of_files_to_transfer)

    cmd = [
        "rclone",
        "copyurl",
        "--urls",
        "copyurls.csv",
        # TODO(Jack): Replace this with dst_root_path
        "/home/jack/data/ICON-EU/grib/rclone_copyurls/",
        # "--progress",
        "--stats=2s",
        # "--quiet",
        # "--stats-one-line",
        "--fast-list",
        "--transfers=16",
        "--checkers=16",
        # "--log-level=ERROR",
        "--stats-log-level=NOTICE",
        "--use-json-log",
    ]
    run_command_with_concurrent_logging(cmd)

    # csv_file.unlink()  # TODO(Jack): uncomment this after testing.


def main() -> None:
    dwd_root_path = PurePosixPath("/weather/nwp/icon-eu/grib/03/")
    dst_root_path = PurePosixPath("/home/jack/data/ICON-EU/grib/rclone_copyurls/")

    # These dwd_paths do not include the the dwd_root_path.

    # dwd_paths = list_grib_files_on_dwd_https(
    #     path=f"dwd-http:{dwd_root_path}",  # remote:path
    # )

    # Find set of unique nwp_init_datetimes.
    # Usually, a DWD path like /weather/nwp/icon-eu/grib/00/ will only contain files for a
    # single NWP init (the midnight init for today). But, if we've listed the HTTPS server while
    # DWD are copying a new NWP run to their FTP server then the 00/ directory will contain
    # files from two NWP init datetimes.

    # unique_nwp_init_datetimes = {
    #     extract_nwp_init_datetime_from_dwd_path(dwd_path) for dwd_path in dwd_paths
    # }
    # log.info(
    #     f"Found {len(unique_nwp_init_datetimes)} unique NWP init datetime(s) in {dwd_root_path}: {unique_nwp_init_datetimes}"
    # )

    # Get a list of all the files in the destination:
    # The paths in this list *include* the NWP init datetime part of the path.

    # files_already_on_dst: list[PurePosixPath] = []
    # for nwp_init_dt in unique_nwp_init_datetimes:
    #     nwp_init_dt_str = format_datetime_for_use_in_dst_path(nwp_init_dt)
    #     files_below_init_dt_folder = list_files(f"{dst_root_path / nwp_init_dt_str}")
    #     files_including_dt_folder = [
    #         nwp_init_dt_str / path for path in files_below_init_dt_folder
    #     ]
    #     files_already_on_dst.extend(files_including_dt_folder)

    # Prepare a CSV with two columns:
    # 1. The source path, from `https://opendata.dwd.de` onwards.
    # 2. The destination path, from the NWP init datetime onwards.
    # This is the format required by `rclone copyurls`.

    csv_of_files_to_transfer: list[str] = []  # Each list item is one line of the CSV.
    # for dwd_path in dwd_paths:
    #     dst_path = convert_dwd_path_to_dst_path(dwd_path)
    #     if dst_path not in files_already_on_dst:
    #         src_path = f"https://opendata.dwd.de{dwd_root_path / dwd_path}"
    #         dst_path = convert_dwd_path_to_dst_path(dwd_path)
    #         csv_of_files_to_transfer.append(f"{src_path},{dst_path}")
    # log.info("Planning to transfer %d files.", len(csv_of_files_to_transfer))

    run_rclone_copyurls("\n".join(csv_of_files_to_transfer))

    # TODO(Jack): Stream stats? Maybe use json stats again? With an update every minute?


if __name__ == "__main__":
    main()
