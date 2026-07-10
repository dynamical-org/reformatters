"""
Copy HRDPS continental GRIB2 files from ECCC's MSC Datamart HTTPS server to Source Co-Op
using `rclone` (see https://rclone.org). The Datamart publishes fully qualified
`{date}/{init_hour}/{lead_hour}` paths, so we `rclone copy` each `{date}/{init_hour}`
directory straight across, preserving the source's own structure, and rely on
`--ignore-existing` for idempotency (Datamart files are immutable once published).
"""

import os
from collections.abc import Sequence
from typing import Any, Final

import pandas as pd

from reformatters.common.logging import get_logger
from reformatters.common.rclone import run_command_with_concurrent_logging
from reformatters.common.retry import retry

log = get_logger(__name__)

MSC_DATAMART_HOST: Final[str] = "https://dd.weather.gc.ca"


def copy_files_from_eccc_https(
    dst_root_path: str,
    nwp_init_hours: Sequence[int],
    days_back: int,
    transfer_parallelism: int,
    checkers: int,
    stats_logging_freq: str,
    env_vars: dict[str, Any] | None = None,
) -> None:
    """
    Args:
        dst_root_path: The destination root directory, in the format `rclone` expects,
            e.g. ':s3:bucket/foo/bar/'. Must end with a forward slash.
        nwp_init_hours: The HRDPS NWP model runs to transfer, e.g. (0, 6, 12, 18).
        days_back: How many additional UTC calendar days before today to recheck,
            beyond today. HRDPS runs are 6-hourly; checking yesterday too catches a
            run whose publication crosses the UTC day boundary, or a cron run that
            was missed or ran slow, without extra bookkeeping.
        transfer_parallelism: Passed to `rclone --transfers`.
        checkers: Passed to `rclone --checkers`.
        stats_logging_freq: The period between each stats log, e.g. "1m".
        env_vars: Additional environment variables to give to `rclone`.
    """
    if not dst_root_path.endswith("/"):
        dst_root_path += "/"

    full_env = os.environ.copy()
    if env_vars:
        full_env.update(env_vars)

    now = pd.Timestamp.now("UTC")
    for day_offset in range(days_back + 1):
        date_str = (now - pd.Timedelta(days=day_offset)).strftime("%Y%m%d")
        for nwp_init_hour in nwp_init_hours:
            src_path = (
                f"/{date_str}/WXO-DD/model_hrdps/continental/2.5km/{nwp_init_hour:02d}"
            )
            dst_path = f"{dst_root_path}{date_str}/{nwp_init_hour:02d}"
            retry(
                lambda src_path=src_path, dst_path=dst_path: _copy_one_init_hour(
                    src_path=src_path,
                    dst_path=dst_path,
                    transfer_parallelism=transfer_parallelism,
                    checkers=checkers,
                    stats_logging_freq=stats_logging_freq,
                    env_vars=full_env,
                ),
                max_attempts=2,
            )


def _copy_one_init_hour(
    src_path: str,
    dst_path: str,
    transfer_parallelism: int,
    checkers: int,
    stats_logging_freq: str,
    env_vars: dict[str, Any],
) -> None:
    cmd = (
        "/usr/bin/rclone",
        "copy",
        f":http:{src_path}",
        dst_path,
        f"--http-url={MSC_DATAMART_HOST}",
        "--ignore-existing",
        "--min-age=1m",  # Ignore files that are so young they might still be mid-upload.
        "--filter=+ *.grib2",
        "--filter=- *",
        "--s3-no-check-bucket",  # Workaround for reformatters issue #428
        "--fast-list",
        f"--transfers={transfer_parallelism:d}",
        f"--checkers={checkers:d}",
        f"--stats={stats_logging_freq}",
        "--stats-log-level=ERROR",  # Output stats to stderr.
        "--quiet",  # Only output logs at error level.
        "--stats-one-line",
    )
    return_code = run_command_with_concurrent_logging(cmd, env_vars=env_vars)
    if return_code == 3:
        log.info(f"Source directory not found (not yet published?): '{src_path}'")
    elif return_code != 0:
        raise RuntimeError(
            f"rclone copy exited with code {return_code} for '{src_path}' -> '{dst_path}'"
        )
