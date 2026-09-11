"""The NOMADS mirror: a cron that copies each hourly HRRR init's files (GRIB and NOAA's
`.idx` sidecar, as published) from NOMADS into `s3://dynamical-noaa-hrrr-nomads-mirror/`
under the same keys NODD uses, minutes before NODD lists them. The 18-hour virtual
dataset reads the mirror when NODD does not have a file yet; see "NOMADS mirror" in docs/virtual_datasets.md.
"""

import re
import struct
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Final, NamedTuple, cast

import obstore
import obstore.store
import pandas as pd
import typer

from reformatters.common import download, kubernetes
from reformatters.common.download import httpx_download_to_disk, httpx_get_text
from reformatters.common.kubernetes import CronJob
from reformatters.common.logging import get_logger
from reformatters.common.operational import OperationalResources
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrFileType
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord
from reformatters.noaa.noaa_grib_index import parse_grib_index_lines
from reformatters.noaa.noaa_utils import NOMADS_RETRY_STATUS_CODES, nomads_rate_limiter

log = get_logger(__name__)

MIRROR_BUCKET: Final = "dynamical-noaa-hrrr-nomads-mirror"
MIRROR_BUCKET_REGION: Final = "us-west-2"
MIRROR_LOCATION_PREFIX: Final = f"s3://{MIRROR_BUCKET}/"
MIRROR_SECRET_NAME: Final = "aws-open-data-icechunk-storage-options-key"  # noqa: S105

MIRRORED_FILE_TYPES: Final[tuple[NoaaHrrrFileType, ...]] = ("sfc", "prs", "nat")
MIRRORED_LEAD_HOURS: Final = range(19)

# GRIB2 section 0: b"GRIB", 2 reserved bytes, discipline, edition, then the message's
# total length as a big endian u64.
_GRIB_SECTION_0_BYTES = 16
_GRIB_END_MARKER = b"7777"


def mirror_key(coord: NoaaHrrrSourceFileCoord) -> str:
    """The mirror object key for coord's data file: identical to its NODD key."""
    return coord.relative_path()


_MIRROR_KEY_PATTERN = re.compile(
    r"hrrr\.(?P<date>\d{8})/conus/hrrr\.t(?P<hour>\d{2})z\.wrf(?P<file_type>sfc|prs|nat)f(?P<lead>\d{2})\.grib2"
)


def parse_mirror_key(
    key: str,
) -> tuple[pd.Timestamp, pd.Timedelta, NoaaHrrrFileType] | None:
    """The (init_time, lead_time, file_type) a data file key names, or None for any
    other key (an index, another product)."""
    match = _MIRROR_KEY_PATTERN.fullmatch(key)
    if match is None:
        return None
    init_time = pd.Timestamp(f"{match['date']}T{match['hour']}:00")
    file_type = cast("NoaaHrrrFileType", match["file_type"])
    return init_time, pd.Timedelta(hours=int(match["lead"])), file_type


def mirror_store(*, write: bool) -> obstore.store.S3Store:
    """Anonymous for reads; for writes, signed with the mounted secret's keys, or the
    ambient AWS credentials outside prod."""
    if not write:
        return download.s3_store(MIRROR_LOCATION_PREFIX, region=MIRROR_BUCKET_REGION)
    secret = kubernetes.load_secret(MIRROR_SECRET_NAME)
    return obstore.store.S3Store(
        MIRROR_BUCKET,
        region=MIRROR_BUCKET_REGION,
        **{
            key: secret[key]
            for key in ("access_key_id", "secret_access_key")
            if key in secret
        },
    )


class MirrorResult(NamedTuple):
    copied: list[str]
    pending: list[str]


class _MirrorFile(NamedTuple):
    coord: NoaaHrrrSourceFileCoord
    is_index: bool

    @property
    def name(self) -> str:
        return self.coord.relative_path().rsplit("/", 1)[-1] + (
            ".idx" if self.is_index else ""
        )

    @property
    def key(self) -> str:
        return mirror_key(self.coord) + (".idx" if self.is_index else "")

    @property
    def url(self) -> str:
        return self.coord.get_url(source="nomads") + (".idx" if self.is_index else "")


def download_from_nomads(url: str) -> Path:
    return httpx_download_to_disk(
        url,
        "noaa-hrrr-nomads",
        rate_limiter=nomads_rate_limiter,
        retry_status_codes=NOMADS_RETRY_STATUS_CODES,
    )


def list_nomads_directory(url: str) -> set[str]:
    """The file names NOMADS lists in one init's directory (one request)."""
    html = httpx_get_text(
        url,
        rate_limiter=nomads_rate_limiter,
        retry_status_codes=NOMADS_RETRY_STATUS_CODES,
    )
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
    poll_interval: timedelta = timedelta(seconds=1),
    list_directory: Callable[[str], set[str]] = list_nomads_directory,
    fetch: Callable[[str], Path] = download_from_nomads,
) -> MirrorResult:
    """Copy one init's files into the mirror as NOMADS publishes them, until every file
    is copied or `deadline` passes.

    Each poll is one NOMADS directory listing, so the poll rate alone is the listing
    rate; every listing and copy goes through the shared NOMADS limiter. A data file
    is copied once it parses as whole GRIB2 messages, and its index only once it
    lists exactly the messages the mirrored data file holds, so an index in the
    mirror describes the data file beside it; anything else is retried next poll.
    """
    files = [
        _MirrorFile(coord, is_index)
        for file_type in file_types
        for lead in lead_hours
        for coord in [_coord(init_time, lead, file_type)]
        for is_index in (False, True)
    ]
    day_prefix = mirror_key(files[0].coord).rsplit("/", 1)[0] + "/"
    in_mirror: dict[str, int] = {
        meta["path"]: meta["size"]
        for batch in obstore.list(mirror, prefix=day_prefix, chunk_size=10_000)
        for meta in batch
    }
    pending = [file for file in files if file.key not in in_mirror]
    directory_url = files[0].url.rsplit("/", 1)[0] + "/"
    offsets_by_key: dict[str, list[int]] = {}
    result = MirrorResult([], [])
    while pending and pd.Timestamp.now("UTC") < deadline:
        poll_start = time.monotonic()
        listed = list_directory(directory_url)
        for file in [f for f in pending if f.name in listed]:
            data_key = mirror_key(file.coord)
            if file.is_index and data_key not in in_mirror:
                continue
            if file.is_index:
                copied = _copy_index(file, mirror, fetch, in_mirror, offsets_by_key)
            else:
                copied = _copy_data(file, mirror, fetch, in_mirror, offsets_by_key)
            if copied:
                pending.remove(file)
                result.copied.append(file.key)
                log.info(f"Mirrored {file.key}")
        time.sleep(
            max(0.0, poll_interval.total_seconds() - (time.monotonic() - poll_start))
        )
    result.pending.extend(file.key for file in pending)
    if pending:
        log.warning(
            f"Deadline reached with {len(pending)} files not mirrored: {result.pending}"
        )
    return result


def _copy_data(
    file: _MirrorFile,
    mirror: obstore.store.ObjectStore,
    fetch: Callable[[str], Path],
    in_mirror: dict[str, int],
    offsets_by_key: dict[str, list[int]],
) -> bool:
    path = fetch(file.url)
    try:
        offsets = grib_message_offsets(path)
        if offsets is None:
            log.warning(f"{file.name} is not whole GRIB2 on NOMADS yet; will retry")
            return False
        with path.open("rb") as data:
            obstore.put(mirror, file.key, data)
        in_mirror[file.key] = path.stat().st_size
        offsets_by_key[file.key] = offsets
        return True
    finally:
        path.unlink()


def _copy_index(
    file: _MirrorFile,
    mirror: obstore.store.ObjectStore,
    fetch: Callable[[str], Path],
    in_mirror: dict[str, int],
    offsets_by_key: dict[str, list[int]],
) -> bool:
    """Copy an index once it lists exactly the messages the mirrored data file holds.
    A data file copied while NOMADS was still appending messages passes the
    whole-GRIB2 check, so on disagreement the data file is fetched again and replaced
    before the index is exposed."""
    data_key = mirror_key(file.coord)
    path = fetch(file.url)
    try:
        index_offsets = [start for start, *_ in parse_grib_index_lines(path)]
        if not index_offsets:
            log.warning(f"{file.name} is empty on NOMADS; will retry")
            return False
        if data_key not in offsets_by_key:
            # Copied by an earlier pod: scan the mirror's copy before trusting the index.
            offsets_by_key[data_key] = _mirrored_message_offsets(mirror, data_key) or []
        if not _index_matches(index_offsets, offsets_by_key[data_key]):
            log.warning(
                f"{file.name} lists messages the mirrored data file lacks; "
                "re-copying the data file"
            )
            data_file = _MirrorFile(file.coord, is_index=False)
            if not _copy_data(data_file, mirror, fetch, in_mirror, offsets_by_key):
                return False
            if not _index_matches(index_offsets, offsets_by_key[data_key]):
                log.warning(
                    f"{file.name} still disagrees with its data file; will retry"
                )
                return False
        with path.open("rb") as data:
            obstore.put(mirror, file.key, data)
        in_mirror[file.key] = path.stat().st_size
        return True
    finally:
        path.unlink()


def _index_matches(index_offsets: list[int], data_offsets: list[int]) -> bool:
    return index_offsets == data_offsets


def _mirrored_message_offsets(
    mirror: obstore.store.ObjectStore, key: str
) -> list[int] | None:
    with tempfile.NamedTemporaryFile(suffix=".grib2") as copy:
        for chunk in obstore.get(mirror, key):
            copy.write(chunk)
        copy.flush()
        return grib_message_offsets(Path(copy.name))


def is_whole_grib2(path: Path) -> bool:
    """Whether the file is a non-empty sequence of complete GRIB2 messages that tile it
    exactly. A file cut inside a message fails; one cut between messages does not,
    which is what the index check at copy time is for."""
    return grib_message_offsets(path) is not None


def grib_message_offsets(path: Path) -> list[int] | None:
    """The start byte of every GRIB2 message in the file, or None if the file is not a
    non-empty sequence of complete edition-2 messages tiling it exactly."""
    size = path.stat().st_size
    offset = 0
    offsets: list[int] = []
    with path.open("rb") as f:
        while offset < size:
            f.seek(offset)
            header = f.read(_GRIB_SECTION_0_BYTES)
            if (
                len(header) < _GRIB_SECTION_0_BYTES
                or header[:4] != b"GRIB"
                or header[7] != 2
            ):
                return None
            (length,) = struct.unpack(">Q", header[8:_GRIB_SECTION_0_BYTES])
            if length < _GRIB_SECTION_0_BYTES or offset + length > size:
                return None
            f.seek(offset + length - len(_GRIB_END_MARKER))
            if f.read(len(_GRIB_END_MARKER)) != _GRIB_END_MARKER:
                return None
            offsets.append(offset)
            offset += length
    return offsets if offsets and offset == size else None


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


class NoaaHrrrNomadsMirror(OperationalResources):
    """Copies each hourly HRRR init's files from NOMADS into the mirror bucket ahead of
    NOAA's AWS copy; the 18-hour virtual dataset reads it when NODD lags."""

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
                ephemeral_storage="4G",
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
            result = mirror_init_time(
                init_time.tz_localize(None), mirror_store(write=True), deadline=deadline
            )
            log.info(
                f"Mirrored {len(result.copied)} files, {len(result.pending)} left to NODD"
            )

    def get_cli(self) -> typer.Typer:
        app = typer.Typer()
        app.command()(self.mirror_gribs)
        return app
