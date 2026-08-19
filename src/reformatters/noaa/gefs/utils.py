import functools
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import pandas as pd

from reformatters.common.download import (
    http_download_to_disk,
    httpx_download_to_disk,
    s3_store,
)
from reformatters.common.iterating import digest
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import replace
from reformatters.common.source_listing import listed_keys_by_prefix
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_CYCLE_COMPLETE_DELAY,
    GEFSDataVar,
    GefsSourceFileCoord,
    get_grib_element,
)
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index
from reformatters.noaa.noaa_utils import (
    NOMADS_RETRY_STATUS_CODES,
    nomads_rate_limiter,
)

log = get_logger(__name__)

GEFS_S3_LOCATION_PREFIX: Final = "s3://noaa-gefs-pds/"
GEFS_S3_BUCKET_REGION: Final = "us-east-1"

type _DownloadFn = Callable[..., Path]


def _s3_key(coord: GefsSourceFileCoord) -> str:
    url = urlparse(coord.get_url())
    assert url.netloc == coord.primary_base_url, (
        f"Only files in {coord.primary_base_url} can be listed, got {coord.get_url()}"
    )
    return url.path.removeprefix("/")


def _s3_directory(coord: GefsSourceFileCoord) -> str:
    return _s3_key(coord).rsplit("/", 1)[0] + "/"


def gefs_published_coords[T: GefsSourceFileCoord](coords: Sequence[T]) -> list[T]:
    """The subset of `coords` whose lead time the source has begun publishing.

    A cycle publishes lead times in order, so within one source directory — which
    holds every ensemble member of one init time and file type — the greatest lead
    time listed is the production frontier. Coords at or below it are kept even when
    that member's own file is not listed yet, so a file that has reached NOMADS but
    not yet the S3 mirror is still reachable via the fallback in `gefs_download_file`.
    Coords of a cycle old enough to have finished are returned without listing.
    """
    settled_before = pd.Timestamp.now() - GEFS_CYCLE_COMPLETE_DELAY
    in_production: dict[str, list[T]] = defaultdict(list)
    for coord in coords:
        if coord.init_time > settled_before:
            in_production[_s3_directory(coord)].append(coord)

    if not in_production:
        return list(coords)

    listed = listed_keys_by_prefix(
        s3_store(GEFS_S3_LOCATION_PREFIX, region=GEFS_S3_BUCKET_REGION),
        sorted(in_production),
    )
    frontiers: dict[str, pd.Timedelta | None] = {
        directory: max(
            (coord.lead_time for coord in group if _s3_key(coord) in listed),
            default=None,
        )
        for directory, group in in_production.items()
    }

    def published(coord: T) -> bool:
        if coord.init_time <= settled_before:
            return True
        frontier = frontiers[_s3_directory(coord)]
        return frontier is not None and coord.lead_time <= frontier

    kept = [coord for coord in coords if published(coord)]
    if len(kept) < len(coords):
        log.info(
            f"Source published through {frontiers}, "
            f"skipping {len(coords) - len(kept)} of {len(coords)} unpublished files"
        )
    return kept


def _index_data_vars(coord: GefsSourceFileCoord) -> Sequence[GEFSDataVar]:
    """coord.data_vars carrying the element names this file's index uses.

    The v12 reforecast labels some messages with a different element name than the
    operational archive, so the index lookup has to rename to match, the same way
    read_data renames to match GRIB band tags.
    """
    return [
        replace(
            var,
            internal_attrs=replace(
                var.internal_attrs,
                grib_element=get_grib_element(var, coord.init_time),
            ),
        )
        for var in coord.data_vars
    ]


def _download_file_from_gefs_source(
    dataset_id: str,
    coord: GefsSourceFileCoord,
    index_url: str,
    source_url: str,
    download: _DownloadFn,
) -> Path:
    idx_local_path = download(index_url, dataset_id, disk_cache=True)

    starts, ends = grib_message_byte_ranges_from_index(
        idx_local_path, _index_data_vars(coord), coord.init_time, coord.lead_time
    )
    vars_suffix = digest(f"{s}-{e}" for s, e in zip(starts, ends, strict=True))
    return download(
        source_url,
        dataset_id,
        byte_ranges=(starts, ends),
        local_path_suffix=f"-{vars_suffix}",
    )


def gefs_download_file(
    dataset_id: str,
    coord: GefsSourceFileCoord,
) -> Path:
    """Download file from GEFS source with retry and fallback to alternative source."""
    try:
        return _download_file_from_gefs_source(
            dataset_id,
            coord,
            coord.get_index_url(),
            coord.get_url(),
            download=http_download_to_disk,
        )
    except FileNotFoundError:
        # if init time is within the last 4 days, try to download from the fallback source (NOMADS)
        if coord.init_time >= pd.Timestamp.now() - pd.Timedelta(days=4):
            return _download_file_from_gefs_source(
                dataset_id,
                coord,
                coord.get_index_url(fallback=True),
                coord.get_fallback_url(),
                download=functools.partial(
                    httpx_download_to_disk,
                    rate_limiter=nomads_rate_limiter,
                    retry_status_codes=NOMADS_RETRY_STATUS_CODES,
                ),
            )
        else:
            raise
