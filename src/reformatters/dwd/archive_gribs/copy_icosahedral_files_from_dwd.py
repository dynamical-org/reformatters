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
so we list the source once, map each path, and `rclone copyurl` the files missing from the
destination.
"""

import re
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any, Final
from urllib.parse import quote

from reformatters.common.logging import get_logger
from reformatters.common.rclone import copy_urls, list_files
from reformatters.common.retry import retry

log = get_logger(__name__)

DWD_HOST: Final[str] = "https://opendata.dwd.de"
SRC_ROOT_PATH: Final[str] = "/weather/nwp/v1/m/icon-eu/p/"
# The listing reads one index page per parameter, level, run and step directory.
LIST_TIMEOUT_SECONDS: Final[float] = 30 * 60

_RUN_DIR_REGEX: Final = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}):00")


def copy_icosahedral_files_from_dwd_https(
    dst_root_path: str,
    nwp_init_hours: Sequence[int],
    level_types: Sequence[int],
    params: Sequence[str],
    transfer_parallelism: int,
    checkers: int,
    stats_logging_freq: str,
    env_vars: dict[str, Any] | None = None,
) -> None:
    """Copy every file of the selected runs still on DWD's server that is missing from
    `dst_root_path`.

    Args:
        dst_root_path: The destination root directory, in the format `rclone` expects,
            e.g. ':s3:bucket/foo/bar/'.
        nwp_init_hours: The ICON-EU runs to copy, e.g. (0, 6, 12, 18).
        level_types: DWD `lvt1` level types to copy in addition to single-level
            parameters: 100 pressure levels, 106 soil levels, 150 model levels.
        params: DWD parameter names to copy, e.g. ("T_2M",). Empty copies all of them.
        transfer_parallelism: Passed to `rclone --transfers`.
        checkers: Passed to `rclone --checkers`.
        stats_logging_freq: The period between each stats log, e.g. "1m".
        env_vars: Environment variables to add to this process's environment for `rclone`.
    """
    if not dst_root_path.endswith("/"):
        dst_root_path += "/"

    def copy_missing_files() -> None:
        src_paths = list_files(
            path=f":http:{SRC_ROOT_PATH}",
            checkers=checkers,
            rclone_args=(
                f"--http-url={DWD_HOST}",
                "--http-no-head",
                "--min-age=1m",  # Ignore files that are so young they might be incomplete.
                *icosahedral_include_filters(nwp_init_hours, level_types, params),
            ),
            env_vars=env_vars,
            timeout_seconds=LIST_TIMEOUT_SECONDS,
        )
        dst_paths = {src_path: icosahedral_dst_path(src_path) for src_path in src_paths}

        already_on_dst: set[PurePosixPath] = set()
        for run_dir in sorted({dst_path.parts[0] for dst_path in dst_paths.values()}):
            already_on_dst.update(
                run_dir / path
                for path in list_files(
                    path=f"{dst_root_path}{run_dir}/",
                    checkers=checkers,
                    env_vars=env_vars,
                )
            )

        # Oldest run first: DWD deletes the oldest run when it publishes the next one.
        to_copy = sorted(
            (
                (f"{DWD_HOST}{SRC_ROOT_PATH}{quote(str(src_path))}", dst_path)
                for src_path, dst_path in dst_paths.items()
                if dst_path not in already_on_dst
            ),
            key=lambda url_and_dst_path: url_and_dst_path[1],
        )
        log.info(f"Planning to copy {len(to_copy):,d} of {len(dst_paths):,d} files.")
        if to_copy:
            copy_urls(
                sources_and_dst_paths=to_copy,
                dst_root_path=dst_root_path,
                transfer_parallelism=transfer_parallelism,
                checkers=checkers,
                stats_logging_freq=stats_logging_freq,
                env_vars=env_vars,
            )

    retry(copy_missing_files, max_attempts=2)


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
