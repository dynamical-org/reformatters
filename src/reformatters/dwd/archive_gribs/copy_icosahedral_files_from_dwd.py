"""
Copy ICON-EU GRIB2 files on DWD's native icosahedral grid from DWD's HTTPS server to a
run-first directory structure using `rclone` (https://rclone.org).

DWD's layout under `https://opendata.dwd.de/weather/nwp/v1/m/icon-eu/p/`:

    <PARAM>/r/<YYYY-MM-DDTHH:MM>/s/<step>.grib2
    <PARAM>/lvt1/<typeOfFirstFixedSurface>/lv1/<level>/r/<YYYY-MM-DDTHH:MM>/s/<step>.grib2

Our layout, relative to the destination root:

    <YYYY-MM-DDTHH>/<PARAM>/<step>.grib2
    <YYYY-MM-DDTHH>/<PARAM>/lvt1/<typeOfFirstFixedSurface>/lv1/<level>/<step>.grib2

`rclone copy` only preserves relative paths and the run sits below the parameter and level,
so we list the source once, map each path, and `rclone copyurl` each run's files that are
missing from the destination.
"""

import re
import time
from collections.abc import Sequence
from datetime import timedelta
from functools import partial
from pathlib import PurePosixPath
from typing import Any, Final
from urllib.parse import quote

import pandas as pd

from reformatters.common.iterating import group_by
from reformatters.common.logging import get_logger
from reformatters.common.rclone import copy_urls, list_files
from reformatters.common.retry import retry

log = get_logger(__name__)

DWD_HOST: Final[str] = "https://opendata.dwd.de"
SRC_ROOT_PATH: Final[str] = "/weather/nwp/v1/m/icon-eu/p/"
LIST_TIMEOUT_SECONDS: Final[float] = 10 * 60
# DWD publishes a run's last file by about init+3h20m. Younger runs may still be publishing,
# and DWD's index pages give rclone no modification times to filter single files by.
MIN_RUN_AGE: Final[timedelta] = timedelta(hours=4)
# DWD keeps about 21 hours of runs, so every selected run younger than this should be listed.
MAX_EXPECTED_RUN_AGE: Final[timedelta] = timedelta(hours=18)

_RUN_DIR_REGEX: Final = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}):00")


def copy_icosahedral_files_from_dwd_https(
    dst_root_path: str,
    nwp_init_hours: Sequence[int],
    level_types: Sequence[int],
    params: Sequence[str],
    time_budget: timedelta,
    transfer_parallelism: int,
    checkers: int,
    stats_logging_freq: str,
    env_vars: dict[str, Any] | None = None,
) -> None:
    """Copy the files missing from `dst_root_path` of every selected run on DWD's server
    that is at least `MIN_RUN_AGE` old, one run at a time, oldest first.

    Raises if DWD lists no files, or if `time_budget` runs out before the listing or a
    run starts. After copying, also raises if a selected run between `MIN_RUN_AGE` and
    `MAX_EXPECTED_RUN_AGE` old is missing, or a run is incomplete (see `incomplete_runs`).

    Args:
        dst_root_path: The destination root directory, in the format `rclone` expects,
            e.g. ':s3:bucket/foo/bar/'.
        nwp_init_hours: The ICON-EU runs to copy, e.g. (0, 6, 12, 18).
        level_types: DWD `lvt1` level types to copy in addition to single-level
            parameters: 100 pressure levels, 106 soil levels, 150 model levels.
        params: DWD parameter names to copy, e.g. ("T_2M",). Empty copies all of them.
        time_budget: How long after this call the listing or a run may still start. A
            run that has started is not interrupted.
        transfer_parallelism: Passed to `rclone --transfers`.
        checkers: Passed to `rclone --checkers`.
        stats_logging_freq: The period between each stats log, e.g. "1m".
        env_vars: Environment variables to add to this process's environment for `rclone`.
    """
    deadline = time.monotonic() + time_budget.total_seconds()

    def raise_if_out_of_time(before: str) -> None:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Stopped before {before}: the {time_budget} time budget ran out."
            )

    if not dst_root_path.endswith("/"):
        dst_root_path += "/"

    raise_if_out_of_time("listing DWD's files")
    src_paths = retry(
        lambda: list_files(
            path=f":http:{SRC_ROOT_PATH}",
            checkers=checkers,
            rclone_args=(
                f"--http-url={DWD_HOST}",
                "--http-no-head",
                *icosahedral_include_filters(nwp_init_hours, level_types, params),
            ),
            env_vars=env_vars,
            timeout_seconds=LIST_TIMEOUT_SECONDS,
        ),
        max_attempts=2,
    )
    if not src_paths:
        raise RuntimeError(f"Found no icosahedral files on {DWD_HOST}{SRC_ROOT_PATH}")

    now = pd.Timestamp.now("UTC")
    newest_run_dir = (now - MIN_RUN_AGE).strftime("%Y-%m-%dT%H")
    expected_run_dirs = {
        init_time.strftime("%Y-%m-%dT%H")
        for init_time in pd.date_range(
            (now - MAX_EXPECTED_RUN_AGE).ceil("h"),
            (now - MIN_RUN_AGE).floor("h"),
            freq="h",
        )
        if init_time.hour in nwp_init_hours
    }
    src_and_dst_paths = sorted(
        ((src_path, icosahedral_dst_path(src_path)) for src_path in src_paths),
        key=lambda src_and_dst_path: src_and_dst_path[1],
    )
    copied_runs: list[list[PurePosixPath]] = []
    # Oldest run first: DWD deletes the oldest run when it publishes the next one.
    for run in group_by(src_and_dst_paths, lambda src_and_dst: src_and_dst[1].parts[0]):
        run_dir = run[0][1].parts[0]
        if run_dir > newest_run_dir:
            log.info(f"Skipping run {run_dir}, which is younger than {MIN_RUN_AGE}.")
            continue

        raise_if_out_of_time(f"run {run_dir}")
        retry(
            partial(
                _copy_run_files_missing_from_dst,
                run,
                dst_root_path=dst_root_path,
                transfer_parallelism=transfer_parallelism,
                checkers=checkers,
                stats_logging_freq=stats_logging_freq,
                env_vars=env_vars,
            ),
            max_attempts=3,
        )
        copied_runs.append([dst_path for _src_path, dst_path in run])

    listed_run_dirs = {run[0].parts[0] for run in copied_runs}
    missing_runs = [
        f"{run_dir} is not on DWD's server"
        for run_dir in sorted(expected_run_dirs - listed_run_dirs)
    ]
    if problems := missing_runs + incomplete_runs(copied_runs, level_types, params):
        raise RuntimeError(
            "Incomplete icosahedral runs on DWD's server: " + "; ".join(problems)
        )


def _copy_run_files_missing_from_dst(
    run: Sequence[tuple[PurePosixPath, PurePosixPath]],
    dst_root_path: str,
    transfer_parallelism: int,
    checkers: int,
    stats_logging_freq: str,
    env_vars: dict[str, Any] | None,
) -> None:
    run_dir = run[0][1].parts[0]
    already_on_dst = set(
        list_files(
            path=f"{dst_root_path}{run_dir}/",
            checkers=checkers,
            env_vars=env_vars,
        )
    )
    to_copy = [
        (f"{DWD_HOST}{SRC_ROOT_PATH}{quote(str(src_path))}", dst_path)
        for src_path, dst_path in run
        if dst_path.relative_to(run_dir) not in already_on_dst
    ]
    log.info(f"Run {run_dir}: {len(to_copy):,d} of {len(run):,d} files to copy.")
    if to_copy:
        copy_urls(
            sources_and_dst_paths=to_copy,
            dst_root_path=dst_root_path,
            transfer_parallelism=transfer_parallelism,
            checkers=checkers,
            stats_logging_freq=stats_logging_freq,
            env_vars=env_vars,
        )


def incomplete_runs(
    dst_paths_by_run: Sequence[Sequence[PurePosixPath]],
    level_types: Sequence[int],
    params: Sequence[str],
) -> list[str]:
    """Describe each run that lacks files another run of its kind (main 00/06/12/18 UTC, or
    intermediate) has, or that lacks any file of an explicitly requested parameter or, when
    every parameter is requested, of single-level parameters or of a requested level type."""
    files_by_run_dir = {
        paths[0].parts[0]: {path.relative_to(path.parts[0]) for path in paths}
        for paths in dst_paths_by_run
    }

    def is_main_run(run_dir: str) -> bool:
        return int(run_dir[-2:]) % 6 == 0

    files_by_kind: dict[bool, set[PurePosixPath]] = {}
    for run_dir, files in files_by_run_dir.items():
        files_by_kind.setdefault(is_main_run(run_dir), set()).update(files)

    problems = []
    for run_dir, files in files_by_run_dir.items():
        if missing := files_by_kind[is_main_run(run_dir)] - files:
            problems.append(
                f"{run_dir} lacks {len(missing):,d} files that other runs of its kind"
                f" have, e.g. {min(missing)}"
            )
        present_params = {file.parts[0] for file in files}
        problems += [
            f"{run_dir} has no {param} files"
            for param in params
            if param not in present_params
        ]
        if not params:
            if not any(len(file.parts) == 2 for file in files):
                problems.append(f"{run_dir} has no single-level files")
            present_level_types = {
                file.parts[2] for file in files if len(file.parts) == 6
            }
            problems += [
                f"{run_dir} has no lvt1/{level_type} files"
                for level_type in level_types
                if str(level_type) not in present_level_types
            ]
    return problems


def icosahedral_include_filters(
    nwp_init_hours: Sequence[int],
    level_types: Sequence[int],
    params: Sequence[str],
) -> list[str]:
    """`rclone` flags that select the plain `.grib2` files of `params` (all if empty), at
    single level and at each of `level_types`, for runs at `nwp_init_hours`. rclone does
    not list directories these rules cannot match."""
    assert nwp_init_hours, "nwp_init_hours must not be empty"
    param = f"{{{','.join(params)}}}" if params else "*"
    hours = ",".join(f"{hour:02d}" for hour in nwp_init_hours)
    run_and_step = f"r/*T{{{hours}}}:00/s/*.grib2"
    level_dirs = ["", *(f"lvt1/{level_type}/lv1/*/" for level_type in level_types)]
    return [f"--include=/{param}/{level_dir}{run_and_step}" for level_dir in level_dirs]


def icosahedral_dst_path(src_path: PurePosixPath) -> PurePosixPath:
    """Map a path relative to DWD's `p/` directory to our run-first path. DWD's parameter
    and level directories are kept verbatim, so distinct source files never collide."""
    match src_path.parts:
        case (_param, "r", run_dir, "s", step):
            pass
        case (_param, "lvt1", _level_type, "lv1", _level, "r", run_dir, "s", step):
            pass
        case _:
            raise ValueError(f"Unexpected DWD icosahedral path: '{src_path}'")
    run = _RUN_DIR_REGEX.fullmatch(run_dir)
    if run is None or not step.endswith(".grib2"):
        raise ValueError(f"Unexpected DWD icosahedral path: '{src_path}'")
    return PurePosixPath(run.group(1), *src_path.parts[:-4], step)
