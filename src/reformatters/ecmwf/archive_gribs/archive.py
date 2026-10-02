"""Archive ECMWF-origin S2S GRIBs from ECDS into a dynamical-controlled bucket.

The bucket is the authoritative source for the reformatter: reading from ECDS at
reformat time would put its request queue back into the write path. A blob is only
published once its full variable x level x member x lead inventory is validated, so
the presence of an archived object means it is complete.
"""

import shutil
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
import requests

from reformatters.common.download import DOWNLOAD_DIR
from reformatters.common.logging import get_logger
from reformatters.common.rclone import copy_local_file, list_files
from reformatters.common.retry import retry

from .ecds_client import EcdsRequest, StateStore, constraints, costing
from .grib_inventory import check_and_index_archived_blob
from .request_shards import ECMWF_ORIGIN, EcdsSelection

log = get_logger(__name__)

ARCHIVE_WORK_DIR: Final[Path] = DOWNLOAD_DIR / "ecmwf-s2s-archive"
# ECDS serves 8 simultaneous requests without error; 4 keeps most of the queue-time
# saving while leaving room for other users of the account.
DEFAULT_CONCURRENT_REQUESTS: Final[int] = 4
DEFAULT_PUBLICATION_POLL_INTERVAL: Final = pd.Timedelta(minutes=2)

type Publication = Literal["unpublished", "partial", "published"]


@dataclass(frozen=True)
class Availability:
    publication: Publication
    # One message per selection key ECDS lacks; empty unless `publication` is "partial".
    missing: tuple[str, ...] = ()


def archive_initialization(
    init_time: pd.Timestamp,
    selections: Sequence[EcdsSelection],
    dst_root_path: str,
    work_dir: Path = ARCHIVE_WORK_DIR,
    api_url: str | None = None,
    checkers: int = 8,
    concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS,
    poll_seconds: float = 30,
    maximum_polls: int = 240,
    env_vars: dict[str, Any] | None = None,
    publication_deadline: pd.Timestamp | None = None,
    publication_poll_interval: pd.Timedelta = DEFAULT_PUBLICATION_POLL_INTERVAL,
) -> bool:
    """Transfer the selections of `init_time` that are not already archived.

    An initialization ECDS has not published by `publication_deadline` is skipped, so
    a caller working backwards through recent initializations reaches the older,
    published ones. Return whether every selection is archived, including those
    already present.

    Args:
        init_time: The initialization to archive. ECMWF S2S initializes at 00 UTC only.
        selections: The ECDS requests covering the initialization, from
            `request_shards.initialization_selections`.
        dst_root_path: Archive root in the form `rclone` expects, e.g.
            `:s3:bucket/ecmwf-s2s-grib/`.
        work_dir: Local scratch for in-flight request state and blobs.
        checkers: Passed to `rclone --checkers` when listing the destination.
        concurrent_requests: How many selections to retrieve at once.
        poll_seconds: Interval between ECDS job status polls.
        maximum_polls: Give up on a job after this many polls.
        env_vars: Environment variables to add to this process's environment for `rclone`.
        publication_deadline: Keep probing ECDS for publication until this UTC time;
            no probe starts after it. None probes once.
        publication_poll_interval: Time between the starts of successive probes.
    """
    assert len(selections) > 0
    init_time_str = format_init_time(init_time)
    dst_init_path = f"{dst_root_path.rstrip('/')}/{init_time_str}"

    archived_file_names = {
        path.name
        for path in list_files(dst_init_path, checkers=checkers, env_vars=env_vars)
    }
    pending = [
        selection
        for selection in selections
        if selection.file_name not in archived_file_names
    ]
    log.info(
        "%d of %d selections still to archive for %s",
        len(pending),
        len(selections),
        init_time_str,
    )
    if not pending:
        return True

    # Checked over every selection, not the pending subset: an initialization only
    # counts as unpublished when ECDS holds none of it.
    availability, seen_partial = _wait_for_publication(
        init_time,
        selections,
        api_url=api_url,
        deadline=publication_deadline,
        poll_interval=publication_poll_interval,
    )
    if availability.publication == "unpublished":
        log.warning(
            "ECDS has not published %s%s, skipping it",
            init_time_str,
            " (an earlier probe saw it partially published)" if seen_partial else "",
        )
        return False
    assert availability.publication == "published", "; ".join(availability.missing)

    def archive_one(selection: EcdsSelection) -> None:
        retry(
            lambda: _archive_one_selection(
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
        list(pool.map(archive_one, pending))
    return True


def check_available(
    init_time: pd.Timestamp,
    selections: Sequence[EcdsSelection],
    api_url: str | None = None,
) -> Availability:
    """Return how much of `init_time` ECDS has published.

    Once it holds every selection, assert that it will accept each selection's size.
    Both endpoints are unauthenticated, so this gate runs before any credentialed
    request is queued. ECDS answers an initialization it does not hold with empty
    constraint values rather than an error, so empty for every selection means the
    initialization is unpublished, while a variable, lead time or level missing from
    only some of them means it is partial.
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
        return Availability("unpublished")

    missing = tuple(
        message
        for selection, valid in available
        for key in ("variable", "leadtime_hour", "level_value")
        if key != "level_value" or selection.level_values
        if (message := _missing(selection, key, valid, init_time))
    )
    if missing:
        return Availability("partial", missing)

    for selection, _ in available:
        cost, limit = costing(selection.inputs(init_time), api_url=api_url)
        assert cost <= limit, (
            f"{selection.file_name} costs {cost:,.0f}, above the ECDS limit of {limit:,.0f}"
        )
        assert cost == selection.cost, (
            f"ECDS costs {selection.file_name} at {cost:,.0f}, not the expected "
            f"{selection.cost:,.0f}; the request cost model has changed"
        )
    return Availability("published")


def format_init_time(init_time: pd.Timestamp) -> str:
    """The archived directory for an initialization, matching the sibling ECMWF IFS ENS archive.

    S2S initializes at 00 UTC only, so the date alone identifies it.
    """
    return init_time.strftime("%Y-%m-%d")


def _archive_one_selection(
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
    started = time.monotonic()
    request.retrieve(
        selection.inputs(init_time),
        target,
        poll_seconds=poll_seconds,
        maximum_polls=maximum_polls,
    )
    retrieved = time.monotonic()
    index_path = check_and_index_archived_blob(
        target,
        variables=set(selection.variables),
        levels=set(selection.level_values),
        ensemble_members=set(selection.ensemble_members),
        lead_time_labels=set(selection.lead_time_labels),
    )
    indexed = time.monotonic()
    # The index lands first so a blob is never visible without the index that reads it.
    copy_local_file(index_path, f"{dst_init_path}/{index_path.name}", env_vars=env_vars)
    copy_local_file(target, f"{dst_init_path}/{selection.file_name}", env_vars=env_vars)
    log.info(
        "Archived %s/%s from ECDS job %s (%d bytes): retrieve %.1f s, "
        "inventory %.1f s, upload %.1f s",
        dst_init_path,
        selection.file_name,
        request.state_store.read().request_id,
        target.stat().st_size,
        retrieved - started,
        indexed - retrieved,
        time.monotonic() - indexed,
    )
    # Kept until here so a retry resumes the in-flight job and partial download.
    shutil.rmtree(selection_work_dir)


def _wait_for_publication(
    init_time: pd.Timestamp,
    selections: Sequence[EcdsSelection],
    api_url: str | None,
    deadline: pd.Timestamp | None,
    poll_interval: pd.Timedelta,
) -> tuple[Availability, bool]:
    """Probe until published, with the last probe starting exactly at `deadline`.

    Return the final observation and whether any probe saw a partial publication.
    """
    probe_started = _utc_now()
    availability = check_available(init_time, selections, api_url=api_url)
    seen_partial = availability.publication == "partial"
    if (
        deadline is None
        or availability.publication == "published"
        or probe_started >= deadline
    ):
        return availability, seen_partial

    wait_started = probe_started
    log.info(
        "Waiting until %s for ECDS to publish %s", deadline, format_init_time(init_time)
    )
    while availability.publication != "published" and probe_started < deadline:
        next_probe = min(probe_started + poll_interval, deadline)
        time.sleep(max(0.0, (next_probe - _utc_now()).total_seconds()))
        probe_started = _utc_now()
        availability = check_available(init_time, selections, api_url=api_url)
        seen_partial |= availability.publication == "partial"

    if availability.publication == "published":
        log.info(
            "ECDS published %s after waiting %s",
            format_init_time(init_time),
            probe_started - wait_started,
        )
    return availability, seen_partial


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now("UTC")


def _missing(
    selection: EcdsSelection,
    key: str,
    valid: dict[str, list[str]],
    init_time: pd.Timestamp,
) -> str | None:
    missing = sorted(set(selection.inputs(init_time)[key]) - set(valid.get(key, [])))
    if not missing:
        return None
    return f"ECDS has no {ECMWF_ORIGIN} {key} {missing} for {selection.file_name}"
