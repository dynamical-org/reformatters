from pathlib import Path
from typing import ClassVar, Literal

import icechunk
import pandas as pd
import pydantic

from reformatters.common.download import s3_download_to_disk, s3_store
from reformatters.common.types import Timedelta
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.hrrr.forecast_virtual_region_job import (
    S3_BUCKET_REGION,
    S3_LOCATION_PREFIX,
    NoaaHrrrForecastVirtualRegionJob,
    NoaaHrrrForecastVirtualSourceFileCoord,
)

CACHE_LOCATION_PREFIX = "s3://dynamical-noaa-hrrr-nomads-cache/"
CACHE_BUCKET_REGION = "us-west-2"

type RefSource = Literal["cache", "archive"]


def hrrr_fast_virtual_chunk_containers() -> tuple[icechunk.VirtualChunkContainer, ...]:
    """Fresh container objects per call; icechunk containers can't be shared
    pydantic defaults."""
    return (
        icechunk.VirtualChunkContainer(
            CACHE_LOCATION_PREFIX,
            icechunk.s3_store(region=CACHE_BUCKET_REGION),
        ),
        icechunk.VirtualChunkContainer(
            S3_LOCATION_PREFIX, icechunk.s3_store(region=S3_BUCKET_REGION)
        ),
    )


class NoaaHrrrForecast48HourVirtualFastSourceFileCoord(
    NoaaHrrrForecastVirtualSourceFileCoord
):
    """One HRRR wrfsfc file, resolved against whichever bucket currently holds it.

    The NOMADS cache and the AWS archive use identical keys, so `source` selects a
    prefix and nothing else; a ref built against one can be repointed at the other.

    Mutable, unlike other coords: the virtual write loop drops coords from its pending
    set by object identity, so discovery has to record where it found a file on the
    coord it was handed rather than on a copy.
    """

    model_config = pydantic.ConfigDict(frozen=False, strict=True)

    source: RefSource = "cache"

    def get_url(self, source: str = "s3") -> str:  # noqa: ARG002 - prefix comes from self.source
        archive_url = super().get_url()
        if self.source == "archive":
            return archive_url
        return CACHE_LOCATION_PREFIX + archive_url.removeprefix(S3_LOCATION_PREFIX)


class NoaaHrrrForecast48HourVirtualFastRegionJob(NoaaHrrrForecastVirtualRegionJob):
    """Reads the NOMADS cache when it holds a file, else the AWS archive.

    NOMADS publishes minutes before the archive, so the cache is the low-latency path;
    falling back to the archive keeps a broken or expired cache from leaving holes and
    lets a backfill run unchanged over inits older than the cache's retention.
    """

    source_file_coord_class: ClassVar[
        type[NoaaHrrrForecast48HourVirtualFastSourceFileCoord]
    ] = NoaaHrrrForecast48HourVirtualFastSourceFileCoord

    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("14h")

    def discover_available(
        self, pending: list[NoaaHrrrForecastVirtualSourceFileCoord]
    ) -> list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]:
        # The write loop drops coords by identity, so mutate and return these objects.
        for coord in pending:
            coord.source = "cache"  # ty: ignore[unresolved-attribute]
        found = discover_available_by_obstore_listing(
            pending,
            store=s3_store(
                CACHE_LOCATION_PREFIX,
                region=CACHE_BUCKET_REGION,
                skip_signature=False,
            ),
            location_prefix=CACHE_LOCATION_PREFIX,
            require_index=True,
        )

        in_cache = {id(coord) for coord, _size in found}
        from_archive = [coord for coord in pending if id(coord) not in in_cache]
        for coord in from_archive:
            coord.source = "archive"  # ty: ignore[unresolved-attribute]
        return found + discover_available_by_obstore_listing(
            from_archive,
            store=s3_store(S3_LOCATION_PREFIX, region=S3_BUCKET_REGION),
            location_prefix=S3_LOCATION_PREFIX,
            require_index=True,
        )

    def _download_index(self, coord: NoaaHrrrForecastVirtualSourceFileCoord) -> Path:
        if coord.get_url().startswith(CACHE_LOCATION_PREFIX):
            return s3_download_to_disk(
                coord.get_index_url(),
                self.dataset_id,
                region=CACHE_BUCKET_REGION,
                skip_signature=False,
            )
        return super()._download_index(coord)
