"""The NOMADS mirror: a cron that copies each hourly HRRR init's files (GRIB and NOAA's
`.idx` sidecar, as published) from NOMADS into an R2 bucket under their NODD keys.
"""

import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Final, get_args

import httpx
import obstore
import obstore.store
import pandas as pd
import typer

from reformatters.common import kubernetes
from reformatters.common.download import httpx_download_to_disk, httpx_get, s3_store
from reformatters.common.grib import grib_message_offsets
from reformatters.common.kubernetes import CronJob
from reformatters.common.logging import get_logger
from reformatters.common.operational import OperationalResources
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrFileType
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord
from reformatters.noaa.noaa_grib_index import parse_grib_index_lines
from reformatters.noaa.noaa_utils import NOMADS_RETRY_STATUS_CODES, nomads_rate_limiter

log = get_logger(__name__)

MIRROR_BUCKET: Final = "noaa-hrrr-nomads-mirror"
# The bucket's public custom domain: what refs into the mirror point at and readers
# fetch from. Listing and writing go through R2's S3 API, which has no anonymous access.
MIRROR_LOCATION_PREFIX: Final = f"https://{MIRROR_BUCKET}.r2.dynamical.org/"
MIRROR_SECRET_NAME: Final = "noaa-hrrr-nomads-mirror-storage-options-key"  # noqa: S105

MIRRORED_FILE_TYPES: Final[tuple[NoaaHrrrFileType, ...]] = ("sfc", "prs", "nat")
MIRRORED_LEAD_HOURS: Final = range(19)
# NOMADS keeps about two days of inits.
CATCH_UP_INIT_TIMES: Final = 24
# Catch-up stops starting copies this long before the fire's deadline, so one in
# flight can finish.
CATCH_UP_DEADLINE_MARGIN: Final = timedelta(minutes=5)
# Listings share the NOMADS rate limiter with downloads, so poll no faster than this.
POLL_INTERVAL: Final = timedelta(seconds=20)
# Bounded by the pod's ephemeral storage: each copy holds one data file on disk.
MAX_CONCURRENT_COPIES: Final = 4


def mirror_store(
    options: Mapping[str, Any] | None = None, bucket: str = MIRROR_BUCKET
) -> obstore.store.S3Store:
    """`bucket` over R2's S3 API, signed with `icechunk.s3_storage` options, by default
    the mounted secret's."""
    if options is None:
        options = kubernetes.load_secret(MIRROR_SECRET_NAME)
    return s3_store(
        f"s3://{bucket}",
        region=options.get("region", "auto"),
        skip_signature=False,
        endpoint=options["endpoint_url"],
        access_key_id=options["access_key_id"],
        secret_access_key=options["secret_access_key"],
        virtual_hosted_style_request=not options.get("force_path_style", False),
    )


class MirrorResult(FrozenBaseModel):
    copied: list[str]
    pending: list[str]


class _PairCopy(FrozenBaseModel):
    """What one task copies of a data file and its index."""

    coord: NoaaHrrrSourceFileCoord
    copy_data: bool
    copy_index: bool
    # Message offsets of the data file already in the mirror, when known.
    data_offsets: list[int] | None

    @property
    def data_key(self) -> str:
        return self.coord.relative_path()

    @property
    def index_key(self) -> str:
        return self.data_key + ".idx"


class _PairResult(FrozenBaseModel):
    copied: list[str]
    data_offsets: list[int] | None


def download_from_nomads(url: str) -> Path:
    return httpx_download_to_disk(
        url,
        "noaa-hrrr-nomads",
        rate_limiter=nomads_rate_limiter,
        retry_status_codes=NOMADS_RETRY_STATUS_CODES,
    )


def list_nomads_directory(url: str) -> set[str]:
    """The file names NOMADS lists in one init's directory (one request)."""
    html = httpx_get(
        url,
        rate_limiter=nomads_rate_limiter,
        retry_status_codes=NOMADS_RETRY_STATUS_CODES,
    ).text
    return set(
        re.findall(r'href="(hrrr\.t\d{2}z\.wrf\w+f\d{2}\.grib2(?:\.idx)?)"', html)
    )


def mirror_init_time(
    init_time: pd.Timestamp,
    mirror: obstore.store.ObjectStore,
    *,
    deadline: pd.Timestamp,
    file_types: Sequence[NoaaHrrrFileType] = MIRRORED_FILE_TYPES,
    lead_hours: Sequence[int] = MIRRORED_LEAD_HOURS,
    poll_interval: timedelta = POLL_INTERVAL,
    max_polls: int | None = None,
    max_concurrent_copies: int = MAX_CONCURRENT_COPIES,
    list_directory: Callable[[str], set[str]] = list_nomads_directory,
    fetch: Callable[[str], Path] = download_from_nomads,
) -> MirrorResult:
    """Copy one init's files into the mirror as NOMADS publishes them, until every file
    is copied, `deadline` passes, or `max_polls` polls have run; copies started before
    then finish before this returns.

    Up to `max_concurrent_copies` data file and index pairs copy in parallel while
    polling continues; every listing and download shares the NOMADS rate limiter. A
    data file is copied once it parses as whole GRIB2 messages, and its index only
    once it lists exactly the messages the mirrored data file holds, so an index in
    the mirror describes the data file beside it; anything else is retried next poll.
    """
    copies = _InitCopies(init_time, mirror, file_types, lead_hours)
    directory_url = copies.coords[0].get_url(source="nomads").rsplit("/", 1)[0] + "/"
    polls = 0
    next_poll = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_concurrent_copies) as pool:
        while copies.pending and pd.Timestamp.now("UTC") < deadline:
            can_poll = max_polls is None or polls < max_polls
            if copies.waiting() and can_poll and time.monotonic() >= next_poll:
                polls += 1
                copies.listed(list_directory(directory_url))
                next_poll = time.monotonic() + poll_interval.total_seconds()
            copies.admit(pool, max_concurrent_copies, mirror, fetch, deadline)
            can_poll = max_polls is None or polls < max_polls
            until_poll = max(0.0, next_poll - time.monotonic())
            if not copies.in_flight:
                if not can_poll:
                    break
                time.sleep(until_poll)
                continue
            # With every pending pair in flight, only a finished copy changes anything.
            done, _ = wait(
                copies.in_flight,
                timeout=until_poll if can_poll and copies.waiting() else None,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                copies.harvest(future)
        for future in as_completed(list(copies.in_flight)):
            copies.harvest(future)
    result = MirrorResult(
        copied=copies.copied, pending=[k for k in copies.keys if k in copies.pending]
    )
    if result.pending:
        log.warning(
            f"{len(result.pending)} files of {init_time:%Y-%m-%dT%H}Z not mirrored"
        )
        log.debug(f"Not mirrored: {result.pending}")
    return result


class _InitCopies:
    """One init's copy state, owned by the polling thread; workers only return results."""

    def __init__(
        self,
        init_time: pd.Timestamp,
        mirror: obstore.store.ObjectStore,
        file_types: Sequence[NoaaHrrrFileType],
        lead_hours: Sequence[int],
    ) -> None:
        self.coords = [
            _coord(init_time, lead, file_type)
            for file_type in file_types
            for lead in lead_hours
        ]
        self.keys = [
            key
            for c in self.coords
            for key in (c.relative_path(), c.relative_path() + ".idx")
        ]
        day_prefix = self.keys[0].rsplit("/", 1)[0] + "/"
        self.in_mirror: set[str] = {
            meta["path"]
            for batch in obstore.list(mirror, prefix=day_prefix, chunk_size=10_000)
            for meta in batch
        }
        self.pending = set(self.keys) - self.in_mirror
        self.copied: list[str] = []
        self.in_flight: dict[Future[_PairResult], _PairCopy] = {}
        self._offsets_by_key: dict[str, list[int]] = {}
        self._listed: set[str] = set()
        # A pair is tried at most once per listing, so a file NOMADS is still writing
        # is not fetched again until the next poll.
        self._tried_since_listing: set[str] = set()

    def waiting(self) -> list[NoaaHrrrSourceFileCoord]:
        """The coords with a file still to copy and no copy in flight."""
        copying = {pair.data_key for pair in self.in_flight.values()}
        return [
            c
            for c in self.coords
            if c.relative_path() not in copying
            and {c.relative_path(), c.relative_path() + ".idx"} & self.pending
        ]

    def listed(self, names: set[str]) -> None:
        self._listed = names
        self._tried_since_listing.clear()

    def admit(
        self,
        pool: ThreadPoolExecutor,
        capacity: int,
        mirror: obstore.store.ObjectStore,
        fetch: Callable[[str], Path],
        deadline: pd.Timestamp,
    ) -> None:
        """Start copies up to `capacity` in flight, so nothing queues past the deadline."""
        for coord in self.waiting():
            if len(self.in_flight) >= capacity:
                return
            if coord.relative_path() in self._tried_since_listing:
                continue
            pair = _pair_copy(
                coord, self._listed, self.pending, self.in_mirror, self._offsets_by_key
            )
            if pair is not None:
                self._tried_since_listing.add(pair.data_key)
                future = pool.submit(_copy_pair, pair, mirror, fetch, deadline)
                self.in_flight[future] = pair

    def harvest(self, future: Future[_PairResult]) -> None:
        pair = self.in_flight.pop(future)
        result = future.result()
        if result.data_offsets is not None:
            self._offsets_by_key[pair.data_key] = result.data_offsets
        for key in result.copied:
            self.pending.discard(key)
            self.in_mirror.add(key)
            self.copied.append(key)
            log.info(f"Mirrored {key}")


def mirror_earlier_init_times(
    init_time: pd.Timestamp,
    mirror: obstore.store.ObjectStore,
    *,
    deadline: pd.Timestamp,
    init_times_back: int = CATCH_UP_INIT_TIMES,
    file_types: Sequence[NoaaHrrrFileType] = MIRRORED_FILE_TYPES,
    lead_hours: Sequence[int] = MIRRORED_LEAD_HOURS,
    max_concurrent_copies: int = MAX_CONCURRENT_COPIES,
    list_directory: Callable[[str], set[str]] = list_nomads_directory,
    fetch: Callable[[str], Path] = download_from_nomads,
) -> MirrorResult:
    """Copy what the mirror lacks of the `init_times_back` inits before `init_time`,
    newest first, one poll each, until `deadline`. An init the mirror holds whole costs
    one mirror listing and no NOMADS request."""
    copied: list[str] = []
    pending: list[str] = []
    for hours_back in range(1, init_times_back + 1):
        if pd.Timestamp.now("UTC") >= deadline:
            log.warning(
                f"Deadline reached with {init_times_back - hours_back + 1} "
                "earlier inits not checked"
            )
            break
        earlier = mirror_init_time(
            init_time - pd.Timedelta(hours=hours_back),
            mirror,
            deadline=deadline,
            file_types=file_types,
            lead_hours=lead_hours,
            max_polls=1,
            max_concurrent_copies=max_concurrent_copies,
            list_directory=list_directory,
            fetch=fetch,
        )
        copied.extend(earlier.copied)
        pending.extend(earlier.pending)
    return MirrorResult(copied=copied, pending=pending)


def _pair_copy(
    coord: NoaaHrrrSourceFileCoord,
    listed: set[str],
    pending: set[str],
    in_mirror: set[str],
    offsets_by_key: dict[str, list[int]],
) -> _PairCopy | None:
    """The copy to start for coord's files given what NOMADS lists, or None. An index
    is only copied beside a data file that is, or is being, mirrored."""
    data_key = coord.relative_path()
    data_name = data_key.rsplit("/", 1)[-1]
    copy_data = data_key in pending and data_name in listed
    copy_index = (
        data_key + ".idx" in pending
        and data_name + ".idx" in listed
        and (copy_data or data_key in in_mirror)
    )
    if not (copy_data or copy_index):
        return None
    return _PairCopy(
        coord=coord,
        copy_data=copy_data,
        copy_index=copy_index,
        data_offsets=offsets_by_key.get(data_key),
    )


def _copy_pair(
    pair: _PairCopy,
    mirror: obstore.store.ObjectStore,
    fetch: Callable[[str], Path],
    deadline: pd.Timestamp,
) -> _PairResult:
    """Copy the pair's data file, then its index once it agrees with the mirrored data
    file. A data file copied while NOMADS was still appending messages passes the
    whole-GRIB2 check, so on disagreement the data file is fetched again and replaced
    before the index is exposed."""
    if _past(deadline):
        return _PairResult(copied=[], data_offsets=None)
    index_path = (
        _fetch_if_served(fetch, pair.coord.get_idx_url(source="nomads"))
        if pair.copy_index
        else None
    )
    try:
        copied: list[str] = []
        data_offsets = pair.data_offsets
        if pair.copy_data:
            data_offsets = _copy_data(pair, mirror, fetch, deadline)
            if data_offsets is None:
                return _PairResult(copied=copied, data_offsets=None)
            copied.append(pair.data_key)
        if index_path is None:
            return _PairResult(copied=copied, data_offsets=data_offsets)
        index_offsets = _index_offsets(index_path)
        if not index_offsets:
            log.warning(
                f"{pair.index_key} is empty or partly written on NOMADS; will retry"
            )
            return _PairResult(copied=copied, data_offsets=data_offsets)
        if data_offsets is None:
            # Copied by an earlier pod: scan the mirror's copy before trusting the index.
            data_offsets = _mirrored_message_offsets(mirror, pair.data_key) or []
        if index_offsets != data_offsets:
            log.warning(
                f"{pair.index_key} lists messages the mirrored data file lacks; "
                "re-copying the data file"
            )
            data_offsets = _copy_data(pair, mirror, fetch, deadline)
            if index_offsets != data_offsets:
                log.warning(
                    f"{pair.index_key} still disagrees with its data file; will retry"
                )
                return _PairResult(copied=copied, data_offsets=data_offsets)
        with index_path.open("rb") as index:
            obstore.put(mirror, pair.index_key, index)
        copied.append(pair.index_key)
        return _PairResult(copied=copied, data_offsets=data_offsets)
    finally:
        if index_path is not None:
            index_path.unlink()


def _copy_data(
    pair: _PairCopy,
    mirror: obstore.store.ObjectStore,
    fetch: Callable[[str], Path],
    deadline: pd.Timestamp,
) -> list[int] | None:
    """Copy the data file if it is whole GRIB2 and `deadline` has not passed, and return
    its message offsets."""
    if _past(deadline):
        return None
    path = _fetch_if_served(fetch, pair.coord.get_url(source="nomads"))
    if path is None:
        return None
    try:
        offsets = grib_message_offsets(path)
        if offsets is None:
            log.warning(f"{pair.data_key} is not whole GRIB2 on NOMADS yet; will retry")
            return None
        with path.open("rb") as data:
            obstore.put(mirror, pair.data_key, data)
        return offsets
    finally:
        path.unlink()


def _fetch_if_served(fetch: Callable[[str], Path], url: str) -> Path | None:
    """The fetched file, or None when NOMADS lists it but does not serve it: a 404, or
    a transfer that fails on every retry, while the file is written or replaced."""
    try:
        return fetch(url)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            raise
    except httpx.TransportError:
        pass
    log.warning(
        f"{url.rsplit('/', 1)[-1]} is listed but not served by NOMADS; will retry"
    )
    return None


def _past(deadline: pd.Timestamp) -> bool:
    return pd.Timestamp.now("UTC") > deadline


def _index_offsets(path: Path) -> list[int]:
    try:
        return [start for start, *_ in parse_grib_index_lines(path)]
    except ValueError, IndexError:
        return []


def _mirrored_message_offsets(
    mirror: obstore.store.ObjectStore, key: str
) -> list[int] | None:
    with tempfile.NamedTemporaryFile(suffix=".grib2") as copy:
        for chunk in obstore.get(mirror, key):
            copy.write(chunk)
        copy.flush()
        return grib_message_offsets(Path(copy.name))


def _coord(
    init_time: pd.Timestamp, lead_hours: int, file_type: NoaaHrrrFileType
) -> NoaaHrrrSourceFileCoord:
    return NoaaHrrrSourceFileCoord(
        init_time=init_time,
        lead_time=pd.Timedelta(hours=lead_hours),
        domain="conus",
        file_type=file_type,
        data_vars=[],
    )


def mirror_window(
    now: pd.Timestamp, cron_job: CronJob, poll_start_minutes: int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(init to mirror, when to start polling, when to stop) for the fire `now` belongs
    to; anchored to the scheduled fire so a replaced pod keeps its predecessor's init."""
    fire = cron_job.previous_fire_time(now)
    init_time = fire.floor("1h")
    return (
        init_time,
        init_time + timedelta(minutes=poll_start_minutes),
        fire + cron_job.pod_active_deadline - timedelta(minutes=1),
    )


def mirror_fire(
    init_time: pd.Timestamp, deadline: pd.Timestamp, mirror: obstore.store.ObjectStore
) -> None:
    """One scheduled fire: mirror `init_time` as NOMADS publishes it, then catch up.
    Raises once catch-up is done if any of `init_time`'s files is still not mirrored,
    which by the deadline means NOMADS was late or failing."""
    result = mirror_init_time(init_time.tz_localize(None), mirror, deadline=deadline)
    log.info(f"Mirrored {len(result.copied)} files, {len(result.pending)} not mirrored")
    caught_up = mirror_earlier_init_times(
        init_time.tz_localize(None),
        mirror,
        deadline=deadline - CATCH_UP_DEADLINE_MARGIN,
    )
    log.info(
        f"Caught up {len(caught_up.copied)} files of earlier inits, "
        f"{len(caught_up.pending)} not mirrored"
    )
    if result.pending:
        raise RuntimeError(
            f"{len(result.pending)} files of {init_time:%Y-%m-%dT%H}Z not mirrored "
            f"by the deadline: {result.pending}"
        )


def mirror_pilot(
    init_time: pd.Timestamp,
    mirror_options: Mapping[str, Any],
    bucket: str,
    file_types: Sequence[str],
    lead_hours: Sequence[int],
    deadline: pd.Timestamp,
    *,
    list_directory: Callable[[str], set[str]] = list_nomads_directory,
    fetch: Callable[[str], Path] = download_from_nomads,
) -> MirrorResult:
    """Copy the chosen files of one published init into a bucket other than the
    mirror's: one NOMADS listing, no catch-up."""
    assert bucket != MIRROR_BUCKET, "A pilot must not write the production mirror"
    allowed = get_args(NoaaHrrrFileType.__value__)
    assert set(file_types) <= set(allowed), f"file types must be among {allowed}"
    return mirror_init_time(
        init_time,
        mirror_store(mirror_options, bucket=bucket),
        deadline=deadline,
        file_types=[t for t in allowed if t in file_types],
        lead_hours=lead_hours,
        max_polls=1,
        list_directory=list_directory,
        fetch=fetch,
    )


class NoaaHrrrNomadsMirror(OperationalResources):
    """Copies each hourly HRRR init's files from NOMADS into the mirror bucket."""

    @property
    def dataset_id(self) -> str:
        return "noaa-hrrr-nomads"

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Hourly with a deadline inside the hour (concurrencyPolicy Replace); sfc f00
        # lands ~init+51m and f18 by ~init+87m, so the fire starts polling at init+49m.
        return [
            CronJob(
                command=["mirror-gribs"],
                workers_total=1,
                parallelism=1,
                name=f"{self.dataset_id}-mirror-gribs",
                schedule="45 * * * *",
                pod_active_deadline=timedelta(minutes=59),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="2G",
                ephemeral_storage="8G",
                secret_names=[MIRROR_SECRET_NAME],
            )
        ]

    def mirror_gribs(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        poll_start_minutes: int = 49,
    ) -> None:
        with self._monitor(
            CronJob, reformat_job_name, cron_job_name=f"{self.dataset_id}-mirror-gribs"
        ):
            now = pd.Timestamp.now("UTC")
            init_time, poll_start, deadline = mirror_window(
                now, self._operational_cron_job(CronJob), poll_start_minutes
            )
            wait = max(0.0, (poll_start - now).total_seconds())
            log.info(
                f"Mirroring {init_time:%Y-%m-%dT%H}Z from {poll_start:%H:%M}Z ({wait:.0f}s)"
            )
            time.sleep(wait)
            mirror_fire(init_time, deadline, mirror_store())

    def get_cli(self) -> typer.Typer:
        app = typer.Typer()
        app.command()(self.mirror_gribs)
        return app
