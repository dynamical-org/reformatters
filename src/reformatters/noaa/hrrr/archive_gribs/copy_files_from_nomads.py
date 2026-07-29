"""Mirror HRRR `wrfsfc` GRIB2 files and their `.idx` sidecars from NOMADS into a cache
bucket with `rclone` (see https://rclone.org).

NOMADS publishes each file minutes before it reaches NOAA's AWS archive, so a virtual
dataset whose refs point at this cache becomes readable sooner. NOMADS keeps only ~2
days: this is a leading-edge cache, not an archive. Cache keys are byte-identical to the
AWS archive's keys so a ref can later be repointed by prefix swap alone.

Unlike the DWD and ECCC archivers this copy sits on the latency path, which drives three
differences: it polls through a cycle's publication window rather than sweeping once,
it passes an explicit `--files-from` list rather than copying directories (NOMADS limits
request count, so per-file source stats must be avoided for files already copied), and
it copies each GRIB2 before its `.idx` so that an index present in the cache implies its
data file is complete.
"""

import re
import tempfile
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, Final

import pandas as pd

from reformatters.common.download import httpx_get_text
from reformatters.common.logging import get_logger
from reformatters.common.rclone import run_command_with_concurrent_logging
from reformatters.noaa.noaa_utils import NOMADS_RETRY_STATUS_CODES, nomads_rate_limiter

log = get_logger(__name__)

NOMADS_HOST: Final[str] = "https://nomads.ncep.noaa.gov"
_HRRR_PROD_PATH: Final[str] = "/pub/data/nccf/com/hrrr/prod"
_HREF_RE: Final[re.Pattern[str]] = re.compile(r'href="([^"/?]+\.grib2(?:\.idx)?)"')


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
    transfer_parallelism: int,
    checkers: int,
    stats_logging_freq: str,
    env_vars: dict[str, Any] | None,
) -> None:
    """Copy every (init_time, lead_hour) wrfsfc file to the cache as NOMADS publishes it.

    Returns once every file has been copied or `max_duration` elapses; a file NOMADS
    never publishes is left behind rather than blocking the rest.
    """
    pending = {
        init_time: {grib_file_name(init_time, lead) for lead in lead_hours}
        for init_time in init_times
    }
    give_up_at = time.monotonic() + max_duration.total_seconds()

    while any(pending.values()):
        poll_start = time.monotonic()
        for init_time, names in pending.items():
            if not names:
                continue
            published = _published_file_names(init_time)
            # An index alone does not mean its data file has landed; require both.
            ready = sorted(
                name
                for name in names
                if name in published and f"{name}.idx" in published
            )
            if not ready:
                continue
            src = source_dir(init_time)
            dst = destination_dir(dst_root_path, init_time)
            for file_names in (ready, [f"{name}.idx" for name in ready]):
                _rclone_copy(
                    src,
                    dst,
                    file_names,
                    transfer_parallelism=transfer_parallelism,
                    checkers=checkers,
                    stats_logging_freq=stats_logging_freq,
                    env_vars=env_vars,
                )
            names.difference_update(ready)
            log.info(
                f"Mirrored {len(ready)} files for {init_time:%Y-%m-%dT%H}Z, "
                f"{len(names)} still pending"
            )

        if not any(pending.values()):
            break
        if time.monotonic() >= give_up_at:
            unmirrored = {
                f"{init_time:%Y-%m-%dT%H}Z": sorted(names)
                for init_time, names in pending.items()
                if names
            }
            log.warning(f"Gave up waiting for unpublished files: {unmirrored}")
            return
        time.sleep(
            max(0.0, poll_interval.total_seconds() - (time.monotonic() - poll_start))
        )


def _published_file_names(init_time: pd.Timestamp) -> set[str]:
    """File names listed in NOMADS' directory index for one init, empty if absent.

    One request per call: NOMADS limits request count, and per-file HEAD bursts trip
    Akamai's bot mitigation.
    """
    url = f"{NOMADS_HOST}{source_dir(init_time)}/"
    html = httpx_get_text(
        url,
        rate_limiter=nomads_rate_limiter,
        retry_status_codes=NOMADS_RETRY_STATUS_CODES,
    )
    return set(_HREF_RE.findall(html))


def _rclone_copy(
    src_path: str,
    dst_path: str,
    file_names: Sequence[str],
    *,
    transfer_parallelism: int,
    checkers: int,
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
            "--multi-thread-cutoff=32M",
            "--multi-thread-streams=4",
            "--s3-no-check-bucket",
            f"--transfers={transfer_parallelism:d}",
            f"--checkers={checkers:d}",
            f"--stats={stats_logging_freq}",
            "--stats-log-level=ERROR",
            "--quiet",
            "--stats-one-line",
        )
        return_code = run_command_with_concurrent_logging(cmd, env_vars=env_vars)
    if return_code != 0:
        raise RuntimeError(
            f"rclone copy exited with code {return_code} for '{src_path}' -> '{dst_path}'"
        )
