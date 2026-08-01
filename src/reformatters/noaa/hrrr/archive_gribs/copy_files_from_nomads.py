"""Mirror HRRR `wrfsfc` GRIB2 files and their `.idx` sidecars from NOMADS into a cache
bucket with `rclone` (see https://rclone.org).

NOMADS publishes each file minutes before it reaches NOAA's AWS archive, so a virtual
dataset whose refs point at this cache becomes readable sooner. NOMADS keeps only ~2
days: this is a leading-edge cache, not an archive. Cache keys are byte-identical to the
AWS archive's keys so a ref can later be repointed by prefix swap alone.

Unlike the DWD and ECCC archivers this copy sits on the latency path, which drives three
differences: it polls deterministic file paths through a cycle's publication window,
it attempts only the next pending file rather than listing or copying directories, and
it copies each GRIB2 before its `.idx` so that an index present in the cache implies its
data file is complete.
"""

import tempfile
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, Final

import pandas as pd

from reformatters.common.logging import get_logger
from reformatters.common.rclone import (
    list_file_sizes,
    run_command_with_concurrent_logging,
)

log = get_logger(__name__)

NOMADS_HOST: Final[str] = "https://nomads.ncep.noaa.gov"
_HRRR_PROD_PATH: Final[str] = "/pub/data/nccf/com/hrrr/prod"


def source_dir(init_time: pd.Timestamp) -> str:
    """The NOMADS path, relative to NOMADS_HOST, holding one init's CONUS files."""
    return f"{_HRRR_PROD_PATH}/hrrr.{init_time:%Y%m%d}/conus"


def destination_dir(dst_root_path: str, init_time: pd.Timestamp) -> str:
    return f"{dst_root_path.rstrip('/')}/hrrr.{init_time:%Y%m%d}/conus"


def grib_file_name(init_time: pd.Timestamp, lead_hour: int) -> str:
    return f"hrrr.t{init_time:%H}z.wrfsfcf{lead_hour:02d}.grib2"


def copy_files_from_nomads(
    dst_root_path: str,
    init_times: Sequence[pd.Timestamp],
    lead_hours: Sequence[int],
    max_duration: timedelta,
    poll_interval: timedelta,
    stats_logging_freq: str,
    env_vars: dict[str, Any] | None,
) -> None:
    """Copy every (init_time, lead_hour) wrfsfc file to the cache as NOMADS publishes it.

    Returns once every file has been copied or `max_duration` elapses; a file NOMADS
    never publishes is left behind rather than blocking the rest.
    """
    pending = {
        init_time: list(
            dict.fromkeys(grib_file_name(init_time, lead) for lead in lead_hours)
        )
        for init_time in init_times
    }
    expected_counts = {init_time: len(names) for init_time, names in pending.items()}
    give_up_at = time.monotonic() + max_duration.total_seconds()

    while any(pending.values()):
        poll_start = time.monotonic()
        for init_time, names in pending.items():
            if not names:
                continue
            src = source_dir(init_time)
            dst = destination_dir(dst_root_path, init_time)
            destination_sizes = list_file_sizes(
                dst,
                rclone_args=("--s3-no-check-bucket",),
                env_vars=env_vars,
            )
            names[:] = [
                name
                for name in names
                if destination_sizes.get(name, 0) == 0
                or destination_sizes.get(f"{name}.idx", 0) == 0
            ]
            if names:
                name = names[0]
                if destination_sizes.get(name, 0) == 0:
                    _rclone_copy(
                        src,
                        dst,
                        [name],
                        stats_logging_freq=stats_logging_freq,
                        env_vars=env_vars,
                    )
                    destination_sizes = list_file_sizes(
                        dst,
                        rclone_args=("--s3-no-check-bucket",),
                        env_vars=env_vars,
                    )

                index_name = f"{name}.idx"
                if (
                    destination_sizes.get(name, 0) > 0
                    and destination_sizes.get(index_name, 0) == 0
                ):
                    _rclone_copy(
                        src,
                        dst,
                        [index_name],
                        stats_logging_freq=stats_logging_freq,
                        env_vars=env_vars,
                    )
                    destination_sizes = list_file_sizes(
                        dst,
                        rclone_args=("--s3-no-check-bucket",),
                        env_vars=env_vars,
                    )

                if (
                    destination_sizes.get(name, 0) > 0
                    and destination_sizes.get(index_name, 0) > 0
                ):
                    names.pop(0)

            mirrored_count = expected_counts[init_time] - len(names)
            log.info(
                f"Mirrored {mirrored_count} files for {init_time:%Y-%m-%dT%H}Z, "
                f"{len(names)} still pending"
            )

        if not any(pending.values()):
            break
        if time.monotonic() >= give_up_at:
            unmirrored = {
                f"{init_time:%Y-%m-%dT%H}Z": names
                for init_time, names in pending.items()
                if names
            }
            log.warning(f"Gave up waiting for unpublished files: {unmirrored}")
            return
        time.sleep(
            max(0.0, poll_interval.total_seconds() - (time.monotonic() - poll_start))
        )


def _rclone_copy(
    src_path: str,
    dst_path: str,
    file_names: Sequence[str],
    *,
    stats_logging_freq: str,
    env_vars: dict[str, Any] | None,
) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as files_from:
        files_from.write("\n".join(file_names) + "\n")
        files_from.flush()
        cmd = (
            "/usr/bin/rclone",
            "copy",
            f":http:{src_path}",
            dst_path,
            f"--http-url={NOMADS_HOST}",
            f"--files-from={files_from.name}",
            # Repairs a wrong-size object, where --ignore-existing would skip it forever.
            "--size-only",
            "--no-traverse",
            "--s3-no-check-bucket",
            "--transfers=1",
            "--checkers=1",
            "--retries=1",
            "--low-level-retries=1",
            f"--stats={stats_logging_freq}",
            "--stats-log-level=ERROR",
            "--quiet",
            "--stats-one-line",
        )
        return_code = run_command_with_concurrent_logging(cmd, env_vars=env_vars)
    if return_code != 0:
        log.warning(
            f"rclone copy exited with code {return_code} for '{src_path}' -> "
            f"'{dst_path}'; the deterministic source path remains pending"
        )
