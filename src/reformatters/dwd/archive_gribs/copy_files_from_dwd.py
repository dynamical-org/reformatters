"""
Restructure DWD GRIB files from DWD's HTTPs server to a timestamped directory structure using `rclone`.

`rclone` is a command-line application to manage files on many different storage systems, including
FTP, cloud object storage, and the local filesystem. See https://rclone.org


## EXAMPLE OF DESIRED TRANSFORMATION:

Example source path (DWD's directory structure for ICON-EU as of January 2026):

    https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/alb_rad/
    icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2


Another example source path (DWD's directory structure for ICON-D2-RUC as of Jan 2026, which might
become the directory structure for ICON-EU in the future[1]):

    https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/ALB_RAD/r/2026-01-14T00:00/s/PT000H00M.grib2


Example destination path on Source Co-Op:

    /2026-01-14T00Z/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2


## WHY DOES THIS PYTHON FILE EXIST? CAN'T WE JUST USE `rclone copy`?

`rclone copy --name-transform` cannot restructure the directory based on the timestamp in the
filename because:

1. rclone processes path segments individually, one by one, from left-to-right.
2. rclone explicitly prohibits adding path separators (`/`) during a name transformation.

For example, when processing the path `00/alb_rad/icon-eu_2026012300.grib2.bz2`, `rclone` will
process '00' first, before it has access to the datetime in the base filename. `rclone` cannot
rename one part of the path based on a subsequent part of the path. This fundamental limitation
persists, no matter if we use `regex=` or `command=` with `rclone copy --name-transform`.

Consequently, rclone cannot dynamically create new directory levels based on filename content.

Instead, we do the name transformation in Python, and pass a CSV file mapping from full source URL
to destination path, using `rclone copyurl --urls <csv_file>`.

## REFERENCES

1. https://www.dwd.de/DE/leistungen/opendata/neuigkeiten/opendata_april2025_1.html
"""

import os
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any

from reformatters.common.logging import get_logger

from .list_files import (
    list_files_on_dst_for_all_nwp_runs_available_from_dwd,
    list_grib_files_on_dwd_https,
)
from .path_conversion import convert_src_path_to_dst_path
from .rclone_copyurl import run_rclone_copyurl

log = get_logger(__name__)


def copy_files_from_dwd_https(
    src_host: str,
    src_root_path: PurePosixPath,
    dst_root_path: PurePosixPath,
    transfer_parallelism: int,
    checkers: int,
    stats_logging_freq: str,
    env_vars: dict[str, Any] | None = None,
) -> None:
    """
    Args:
        src_host: The HTTP or HTTPS URL, e.g. "https://opendata.dwd.de".
            Should not include a trailing slash.
        src_root_path: The absolute path on src_host for one NWP run.
            Must start with a forward slash, e.g. "/weather/nwp/icon-eu/grib/00/"
        dst_root_path: The destination path, e.g. "/data/ICON-EU/" or ":s3:bucket/path".
            Must be in the format that rclone expects.
        transfer_parallelism: Number of concurrent workers during the copy operation.
            Each worker fetches a file from src_host, copies it to the destination, and waits for
            the destination to acknowledge completion before fetching another file from the source.
            When fetching from HTTPS and writing to object storage, this could be set arbitrarily
            high, although setting it too high (>256?) might be detrimental to performance.
        checkers: This number is passed to the `rclone --checkers` argument.
            In the context of recursive file listing, it appears `checkers` controls the number of
            directories that are listed in parallel. Note that more is not always better. For
            example, on a small VM with only 2 CPUs, `rclone` maxes out the CPUs if `checkers` is
            above 32, and this actually slows down file listing.
            For more info, see the rclone docs: https://rclone.org/docs/#checkers-int
        stats_logging_freq: The period between each stats log. e.g. "1m" to log stats every minute.
            See https://rclone.org/docs/#stats-duration
        env_vars: Additional environment variables to give to `rclone`. For example:
            {
                "RCLONE_S3_PROVIDER": "AWS",
                "RCLONE_S3_ACCESS_KEY_ID": "key",
                "RCLONE_S3_SECRET_ACCESS_KEY": "secret",
                "RCLONE_S3_REGION": "us-west-2",
                "RCLONE_S3_FORCE_PATH_STYLE": "false",
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


def compute_which_files_still_need_to_be_transferred(
    src_paths_starting_with_nwp_var: Sequence[PurePosixPath],
    files_already_on_dst: set[PurePosixPath],
    src_host_and_root_path: str,
) -> list[str]:
    """Returns list of strings, each of which is a row of a CSV with two columns:

    1. The full source path, e.g. `https://opendata.dwd.de/.../filename.grib2.bz2`.
    2. The destination path, from the NWP init datetime onwards.

    This is the format required by `rclone copyurls`.

    Args
        src_paths_starting_with_nwp_var: Paths must not start with a forwards slash.
        files_already_on_dst:
        src_host_and_root_path: Must not end with forwards slash.
    """
    if src_host_and_root_path[-1] == "/":
        raise ValueError(
            f"src_host_and_root_path must not end with a slash. {src_host_and_root_path=}"
        )

    csv_of_files_to_transfer: list[str] = []  # Each list item is one line of the CSV.
    for i, src_path in enumerate(src_paths_starting_with_nwp_var):
        if src_path.is_absolute():
            raise ValueError(
                "src_paths_starting_with_nwp_var must not start with a slash."
                f" src_paths_starting_with_nwp_var[{i}]='{src_paths_starting_with_nwp_var}'"
            )
        dst_path = convert_src_path_to_dst_path(src_path)
        if dst_path not in files_already_on_dst:
            full_src_path = f"{src_host_and_root_path}/{src_path}"
            csv_of_files_to_transfer.append(f"{full_src_path},{dst_path}")
    log.info(f"Planning to transfer {len(csv_of_files_to_transfer):,d} files.")
    return csv_of_files_to_transfer
