from collections.abc import Sequence
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from reformatters.common.logging import get_logger
from reformatters.common.rclone import list_files

from .path_conversion import (
    extract_nwp_init_datetime_from_grib_filename,
    format_datetime_for_dst_path,
)

log = get_logger(__name__)


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
    for nwp_init_dt in sorted(unique_nwp_init_datetimes):
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
