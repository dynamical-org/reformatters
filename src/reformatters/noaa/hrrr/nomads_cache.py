"""The NOMADS cache: an S3 bucket the mirror cron fills with HRRR GRIB files and their
`wgrib2 -s` indexes minutes before NOAA's AWS archive (NODD) publishes them, under the
same keys NODD uses. See "NOMADS cache" in docs/virtual_datasets.md.
"""

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Final, cast

import obstore
import obstore.store
import pandas as pd
import s3fs

from reformatters.common import download, kubernetes
from reformatters.common.validation import (
    ValidationContext,
    ValidationResult,
    Validator,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrFileType
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord

NOMADS_CACHE_BUCKET: Final = "dynamical-noaa-hrrr-nomads"
NOMADS_CACHE_BUCKET_REGION: Final = "us-west-2"
NOMADS_CACHE_LOCATION_PREFIX: Final = f"s3://{NOMADS_CACHE_BUCKET}/"
NOMADS_CACHE_SECRET_NAME: Final = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
# Written beside a cached file once main's refs for it point at NODD again.
REPOINTED_MARKER_SUFFIX: Final = ".repointed"
# The bucket lifecycle expires only objects carrying this tag.
REPOINTED_TAG: Final = {"repointed": "true"}

_CACHE_KEY_PATTERN = re.compile(
    r"hrrr\.(?P<date>\d{8})/conus/hrrr\.t(?P<hour>\d{2})z\.wrf(?P<file_type>sfc|prs|nat)f(?P<lead>\d{2})\.grib2"
)


def cache_key(coord: NoaaHrrrSourceFileCoord) -> str:
    """The cache object key for coord's data file: identical to its NODD key."""
    return coord.relative_path()


def parse_cache_key(
    key: str,
) -> tuple[pd.Timestamp, pd.Timedelta, NoaaHrrrFileType] | None:
    """The (init_time, lead_time, file_type) a data file key names, or None for any
    other key (an index, a marker)."""
    match = _CACHE_KEY_PATTERN.fullmatch(key)
    if match is None:
        return None
    init_time = pd.Timestamp(f"{match['date']}T{match['hour']}:00")
    file_type = cast("NoaaHrrrFileType", match["file_type"])
    return init_time, pd.Timedelta(hours=int(match["lead"])), file_type


def nomads_cache_store(*, write: bool) -> obstore.store.S3Store:
    """Anonymous for reads; for writes, signed with the mounted secret's keys, or the
    ambient AWS credentials outside prod."""
    if not write:
        return download.s3_store(
            NOMADS_CACHE_LOCATION_PREFIX, region=NOMADS_CACHE_BUCKET_REGION
        )
    return obstore.store.S3Store(
        NOMADS_CACHE_BUCKET, region=NOMADS_CACHE_BUCKET_REGION, **_write_credentials()
    )


def list_cache(store: obstore.store.ObjectStore) -> dict[str, datetime]:
    """Every key in the cache and when it was written."""
    return {
        meta["path"]: meta["last_modified"]
        for batch in obstore.list(store, chunk_size=10_000)
        for meta in batch
    }


def unrepointed_data_files(listing: Mapping[str, datetime]) -> list[str]:
    """Keys of the cached data files whose index exists and whose repointed marker does
    not: main may still reference these in the cache."""
    return sorted(
        key
        for key in listing
        if parse_cache_key(key) is not None
        and key + ".idx" in listing
        and key + REPOINTED_MARKER_SUFFIX not in listing
    )


def mark_repointed(key: str, store: obstore.store.ObjectStore) -> None:
    """Record that main's refs for the cached data file `key` point at NODD: tag the
    file and its index so the bucket lifecycle may expire them, then write the marker
    (tagged too, so it expires alongside). The marker is last: a run that dies between
    the two leaves an unmarked file, which the next fire repoints and marks again.
    """
    is_s3 = isinstance(store, obstore.store.S3Store)
    if is_s3:
        _tag_repointed([key, key + ".idx"])
    obstore.put(
        store, key + REPOINTED_MARKER_SUFFIX, b"", tags=REPOINTED_TAG if is_s3 else None
    )


def _tag_repointed(keys: Sequence[str]) -> None:
    credentials = _write_credentials()
    fs = s3fs.S3FileSystem(
        key=credentials.get("access_key_id"),
        secret=credentials.get("secret_access_key"),
        client_kwargs={"region_name": NOMADS_CACHE_BUCKET_REGION},
    )
    for key in keys:
        fs.put_tags(f"{NOMADS_CACHE_BUCKET}/{key}", dict(REPOINTED_TAG))


def _write_credentials() -> dict[str, str]:
    secret = kubernetes.load_secret(NOMADS_CACHE_SECRET_NAME)
    return {
        key: secret[key]
        for key in ("access_key_id", "secret_access_key")
        if key in secret
    }


class CheckNomadsCacheRepointed(Validator):
    """Fail when a cached data file older than `max_age` has not been repointed to NODD:
    NODD has not published it, so main still depends on the cache for it."""

    max_age: timedelta = timedelta(hours=3)

    def check(self, context: ValidationContext) -> ValidationResult:  # noqa: ARG002 - the cache, not the store, is checked
        listing = list_cache(self.cache_store())
        now = pd.Timestamp.now("UTC")
        stale = [
            key
            for key in unrepointed_data_files(listing)
            if now - listing[key] > self.max_age
        ]
        if stale:
            return ValidationResult(
                passed=False,
                message=f"{len(stale)} cached files older than {self.max_age} still "
                f"point main at the cache (NODD has not published them): {stale[:10]}",
                checked_count=len(listing),
            )
        return ValidationResult(
            passed=True,
            message=f"{len(listing)} cache objects, none awaiting repoint past {self.max_age}",
            checked_count=len(listing),
        )

    def cache_store(self) -> obstore.store.ObjectStore:
        return nomads_cache_store(write=False)
