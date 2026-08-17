"""Stage ECMWF-origin S2S GRIBs from ECDS into a dynamical-controlled bucket.

The bucket is the authoritative source for the reformatter: reading from ECDS at
reformat time would put its request queue back into the write path. A blob is only
published once its full variable x level x member x lead inventory is validated, so
the presence of a staged object means it is complete.
"""

import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests

from reformatters.common.download import DOWNLOAD_DIR
from reformatters.common.logging import get_logger
from reformatters.common.rclone import copy_local_file, list_files
from reformatters.common.retry import retry

from .ecds_client import EcdsRequest, StateStore, constraints, costing
from .grib_inventory import check_and_index_staged_blob
from .request_shards import ECMWF_ORIGIN, EcdsSelection

log = get_logger(__name__)

STAGING_WORK_DIR: Final[Path] = DOWNLOAD_DIR / "ecmwf-s2s-staging"
# ECDS serves 8 simultaneous requests without error; 4 keeps most of the queue-time
# saving while leaving room for other users of the account.
DEFAULT_CONCURRENT_REQUESTS: Final[int] = 4


def stage_initialization(
    init_time: pd.Timestamp,
    selections: Sequence[EcdsSelection],
    dst_root_path: str,
    work_dir: Path = STAGING_WORK_DIR,
    api_url: str | None = None,
    checkers: int = 8,
    concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS,
    poll_seconds: float = 30,
    maximum_polls: int = 240,
    env_vars: dict[str, Any] | None = None,
) -> None:
    """Transfer the selections of `init_time` that are not already staged.

    An initialization ECDS has not published yet is skipped, so a caller working
    backwards through recent initializations reaches the older, published ones.

    Args:
        init_time: The initialization to stage. ECMWF S2S initializes at 00 UTC only.
        selections: The ECDS requests covering the initialization, from
            `request_shards.initialization_selections`.
        dst_root_path: Staging root in the form `rclone` expects, e.g.
            `:s3:bucket/ecmwf-s2s-grib/`.
        work_dir: Local scratch for in-flight request state and blobs.
        checkers: Passed to `rclone --checkers` when listing the destination.
        concurrent_requests: How many selections to retrieve at once.
        poll_seconds: Interval between ECDS job status polls.
        maximum_polls: Give up on a job after this many polls.
        env_vars: Environment variables to add to this process's environment for `rclone`.
    """
    assert len(selections) > 0
    init_time_str = format_init_time(init_time)
    dst_init_path = f"{dst_root_path.rstrip('/')}/{init_time_str}"

    staged_file_names = {
        path.name
        for path in list_files(dst_init_path, checkers=checkers, env_vars=env_vars)
    }
    pending = [
        selection
        for selection in selections
        if selection.file_name not in staged_file_names
    ]
    log.info(
        "%d of %d selections still to stage for %s",
        len(pending),
        len(selections),
        init_time_str,
    )
    if not pending:
        return

    # Checked over every selection, not the pending subset: an initialization only
    # counts as unpublished when ECDS holds none of it.
    if not check_available(init_time, selections, api_url=api_url):
        log.warning("ECDS has not published %s, skipping it", init_time_str)
        return

    def stage_one(selection: EcdsSelection) -> None:
        retry(
            lambda: _stage_one_selection(
                init_time=init_time,
                selection=selection,
                dst_init_path=dst_init_path,
                work_dir=work_dir / init_time_str,
                api_url=api_url,
                poll_seconds=poll_seconds,
                maximum_polls=maximum_polls,
                env_vars=env_vars,
            ),
            max_attempts=3,
            # Transient only: a deterministic inventory AssertionError would re-download
            # a multi-GB blob to fail identically.
            retryable_exceptions=(requests.RequestException, RuntimeError),
        )

    with ThreadPoolExecutor(concurrent_requests) as pool:
        list(pool.map(stage_one, pending))


def check_available(
    init_time: pd.Timestamp,
    selections: Sequence[EcdsSelection],
    api_url: str | None = None,
) -> bool:
    """Return whether ECDS has published `init_time` at all.

    Once it has, assert that it holds every selection and will accept each
    selection's size. Both endpoints are unauthenticated, so this gate runs before
    any credentialed request is queued. ECDS answers an initialization it does not
    hold with empty constraint values rather than an error, so empty for every
    selection means the initialization is not published yet, while empty for only
    some of them means a variable, lead time or level we expect is genuinely missing.
    """
    available = [
        (
            selection,
            constraints(
                {
                    key: value
                    for key, value in selection.inputs(init_time).items()
                    if key not in {"leadtime_hour", "level_value", "data_format"}
                },
                api_url=api_url,
            ),
        )
        for selection in selections
    ]
    if all(not valid.get("variable") for _, valid in available):
        return False

    for selection, valid in available:
        inputs = selection.inputs(init_time)
        _assert_available(selection, "variable", inputs["variable"], valid)
        _assert_available(selection, "leadtime_hour", inputs["leadtime_hour"], valid)
        if selection.level_values:
            _assert_available(selection, "level_value", inputs["level_value"], valid)

        cost, limit = costing(inputs, api_url=api_url)
        assert cost <= limit, (
            f"{selection.file_name} costs {cost:,.0f}, above the ECDS limit of {limit:,.0f}"
        )
        assert cost == selection.cost, (
            f"ECDS costs {selection.file_name} at {cost:,.0f}, not the expected "
            f"{selection.cost:,.0f}; the request cost model has changed"
        )
    return True


def format_init_time(init_time: pd.Timestamp) -> str:
    """The staged directory for an initialization, matching the sibling ECMWF IFS ENS archive.

    S2S initializes at 00 UTC only, so the date alone identifies it.
    """
    return init_time.strftime("%Y-%m-%d")


def _stage_one_selection(
    init_time: pd.Timestamp,
    selection: EcdsSelection,
    dst_init_path: str,
    work_dir: Path,
    api_url: str | None,
    poll_seconds: float,
    maximum_polls: int,
    env_vars: dict[str, Any] | None,
) -> None:
    selection_work_dir = work_dir / selection.file_name
    target = selection_work_dir / selection.file_name
    request = EcdsRequest(
        StateStore(selection_work_dir / "request_state.json"), api_url=api_url
    )
    request.retrieve(
        selection.inputs(init_time),
        target,
        poll_seconds=poll_seconds,
        maximum_polls=maximum_polls,
    )
    index_path = check_and_index_staged_blob(
        target,
        variables=set(selection.variables),
        levels=set(selection.level_values),
        ensemble_members=set(selection.ensemble_members),
        lead_time_labels=set(selection.lead_time_labels),
    )
    # The index lands first so a blob is never visible without the index that reads it.
    copy_local_file(index_path, f"{dst_init_path}/{index_path.name}", env_vars=env_vars)
    copy_local_file(target, f"{dst_init_path}/{selection.file_name}", env_vars=env_vars)
    log.info("Staged %s/%s", dst_init_path, selection.file_name)
    # Kept until here so a retry resumes the in-flight job and partial download.
    shutil.rmtree(selection_work_dir)


def _assert_available(
    selection: EcdsSelection,
    key: str,
    requested: Sequence[str],
    valid: dict[str, list[str]],
) -> None:
    missing = sorted(set(requested) - set(valid.get(key, [])))
    assert not missing, (
        f"ECDS has no {ECMWF_ORIGIN} {key} {missing} for {selection.file_name}"
    )
