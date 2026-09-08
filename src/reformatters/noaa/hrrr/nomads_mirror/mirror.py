import time
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from subprocess import CalledProcessError
from typing import Final, NamedTuple

import obstore
import obstore.store
import pandas as pd

from reformatters.common.download import httpx_download_to_disk, is_not_found
from reformatters.common.logging import get_logger
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrFileType
from reformatters.noaa.hrrr.nomads_cache import cache_key
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord
from reformatters.noaa.noaa_grib_index import (
    parse_grib_index_lines,
    scan_grib_message_offsets,
    write_grib_index,
)
from reformatters.noaa.noaa_utils import NOMADS_RETRY_STATUS_CODES, nomads_rate_limiter

log = get_logger(__name__)
MIRRORED_LEAD_HOURS: Final[Mapping[NoaaHrrrFileType, Sequence[int]]] = {
    "sfc": tuple(range(19))
}


class MirrorResult(NamedTuple):
    mirrored: list[str]
    skipped: list[str]


def download_from_nomads(url: str, retry_timeout: float) -> Path:
    return httpx_download_to_disk(
        url,
        "noaa-hrrr-nomads-cache",
        rate_limiter=nomads_rate_limiter,
        retry_status_codes=NOMADS_RETRY_STATUS_CODES,
        retry_timeout=retry_timeout,
    )


def mirror_init_time(
    init_time: pd.Timestamp,
    cache: obstore.store.ObjectStore,
    *,
    deadline: pd.Timestamp,
    lead_hours: Mapping[NoaaHrrrFileType, Sequence[int]] = MIRRORED_LEAD_HOURS,
    poll_interval: timedelta = timedelta(seconds=5),
    frontier_probe_every: int = 12,
    frontier_width: int = 2,
    max_invalid_attempts: int = 3,
    stop_margin: timedelta = timedelta(minutes=2),
    download: Callable[[str, float], Path] = download_from_nomads,
    build_index: Callable[[Path, Path], None] = write_grib_index,
) -> MirrorResult:
    listed = {
        entry["path"]
        for batch in obstore.list(cache, prefix=f"hrrr.{init_time:%Y%m%d}/conus/")
        for entry in batch
    }
    coords = [
        NoaaHrrrSourceFileCoord(
            init_time=init_time,
            lead_time=pd.Timedelta(hours=lead),
            domain="conus",
            file_type=file_type,
            data_vars=[],
        )
        for file_type, leads in lead_hours.items()
        for lead in sorted(leads)
    ]
    pending = {
        file_type: [
            coord
            for coord in coords
            if coord.file_type == file_type and cache_key(coord) + ".idx" not in listed
        ]
        for file_type in lead_hours
    }
    result = MirrorResult([], [])
    invalid_attempts: dict[str, int] = {}
    previous_counts: dict[NoaaHrrrFileType, int] = {}
    poll = 0
    look_ahead = 0
    while any(pending.values()) and pd.Timestamp.now("UTC") < deadline - stop_margin:
        started = time.monotonic()
        poll += 1
        frontier_poll = poll % frontier_probe_every == 0
        if frontier_poll:
            look_ahead += frontier_width
        for file_type, coords in pending.items():
            missing: list[NoaaHrrrSourceFileCoord] = []
            attempts = (
                frontier_attempts(coords, look_ahead, frontier_width)
                if frontier_poll
                else coords[:1]
            )
            for coord in attempts:
                remaining = (deadline - pd.Timestamp.now("UTC")).total_seconds()
                if remaining <= stop_margin.total_seconds():
                    return result
                key = cache_key(coord)
                path, unpublished = _download_published(
                    coord,
                    download,
                    retry_timeout=remaining - stop_margin.total_seconds(),
                )
                if path is None:
                    if unpublished:
                        missing.append(coord)
                    continue
                count = _validate_and_upload(
                    path,
                    cache,
                    key,
                    previous_counts.get(file_type, 0),
                    build_index,
                    started,
                )
                if count is None:
                    invalid_attempts[key] = invalid_attempts.get(key, 0) + 1
                    if invalid_attempts[key] >= max_invalid_attempts:
                        coords.remove(coord)
                        result.skipped.append(key)
                    continue
                previous_counts[file_type] = count
                coords.remove(coord)
                result.mirrored.append(key)
                look_ahead = 0
                if missing:
                    _skip_lower_unpublished(coords, coord, result)
                    missing.clear()
        if any(pending.values()):
            time.sleep(
                max(0, poll_interval.total_seconds() - (time.monotonic() - started))
            )
    return result


def _skip_lower_unpublished(
    coords: list[NoaaHrrrSourceFileCoord],
    published: NoaaHrrrSourceFileCoord,
    result: MirrorResult,
) -> None:
    """Leave to NODD every pending lead below one NOMADS has published past."""
    for lower in [c for c in coords if c.lead_time < published.lead_time]:
        coords.remove(lower)
        result.skipped.append(cache_key(lower))
        log.warning(f"Skipping unpublished frontier {cache_key(lower)}")


def _download_published(
    coord: NoaaHrrrSourceFileCoord,
    download: Callable[[str, float], Path],
    *,
    retry_timeout: float,
) -> tuple[Path | None, bool]:
    """(downloaded file or None, whether NOMADS answered that it has no such file).
    A failed request is (None, False): the downloader has retried within
    `retry_timeout` seconds, and the next poll tries again."""
    try:
        return download(coord.get_url(source="nomads"), retry_timeout), False
    except Exception as exc:  # noqa: BLE001
        if is_not_found(exc):
            return None, True
        log.warning(f"Download failed for {cache_key(coord)}: {exc}")
        return None, False


def frontier_attempts[T](pending: Sequence[T], look_ahead: int, width: int) -> list[T]:
    """The lowest pending lead plus a `width`-wide window that walks up the list as
    `look_ahead` grows, wrapping to the front, so a run of unpublished leads longer
    than `width` is eventually passed."""
    if not pending:
        return []
    later = pending[1:]
    if not later:
        return [pending[0]]
    start = (look_ahead - width) % len(later)
    return [pending[0], *later[start : start + width]]


def _validate_and_upload(
    path: Path,
    cache: obstore.store.ObjectStore,
    key: str,
    previous_count: int,
    build_index: Callable[[Path, Path], None],
    started: float,
) -> int | None:
    index_path = path.with_suffix(path.suffix + ".idx")
    try:
        try:
            offsets = scan_grib_message_offsets(path)
            build_index(path, index_path)
            lines = parse_grib_index_lines(index_path)
            if [start for start, *_ in lines] != offsets:
                raise ValueError("Inventory offsets do not match GRIB messages")
            if len(lines) < previous_count:
                raise ValueError("Message count is lower than the previous lead")
        except (ValueError, CalledProcessError) as exc:
            log.warning(f"Invalid GRIB for {key}: {exc}")
            return None
        size = path.stat().st_size
        with path.open("rb") as data:
            obstore.put(cache, key, data)
        with index_path.open("rb") as index:
            obstore.put(cache, key + ".idx", index)
        log.info(
            f"Mirrored {key}: {size} bytes in {time.monotonic() - started:.2f} seconds"
        )
        return len(lines)
    finally:
        path.unlink()
        index_path.unlink(missing_ok=True)
