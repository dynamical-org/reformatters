import pandas as pd
import pytest

from reformatters.noaa.hrrr.forecast_48_hour_virtual_fast import region_job as module
from reformatters.noaa.hrrr.forecast_48_hour_virtual_fast.region_job import (
    CACHE_LOCATION_PREFIX,
    NoaaHrrrForecast48HourVirtualFastRegionJob,
    NoaaHrrrForecast48HourVirtualFastSourceFileCoord,
    RefSource,
    hrrr_fast_virtual_chunk_containers,
)
from reformatters.noaa.hrrr.forecast_virtual_region_job import S3_LOCATION_PREFIX

CoordT = NoaaHrrrForecast48HourVirtualFastSourceFileCoord

ARCHIVE_KEY = "hrrr.20240601/conus/hrrr.t00z.wrfsfcf06.grib2"


def make_coord(
    source: RefSource = "cache",
) -> NoaaHrrrForecast48HourVirtualFastSourceFileCoord:
    coord = NoaaHrrrForecast48HourVirtualFastSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T00:00"),
        lead_time=pd.Timedelta(hours=6),
        domain="conus",
        file_type="sfc",
        data_vars=[],
    )
    coord.source = source
    return coord


def test_cache_and_archive_urls_share_one_key() -> None:
    """Identical keys are what let a ref be repointed between buckets by prefix
    swap alone."""
    cache_url = make_coord("cache").get_url()
    archive_url = make_coord("archive").get_url()
    assert cache_url == CACHE_LOCATION_PREFIX + ARCHIVE_KEY
    assert archive_url == S3_LOCATION_PREFIX + ARCHIVE_KEY
    assert cache_url.removeprefix(CACHE_LOCATION_PREFIX) == archive_url.removeprefix(
        S3_LOCATION_PREFIX
    )


def test_index_url_follows_the_selected_source() -> None:
    assert make_coord("cache").get_index_url().startswith(CACHE_LOCATION_PREFIX)
    assert make_coord("archive").get_index_url().startswith(S3_LOCATION_PREFIX)


def test_containers_cover_both_prefixes() -> None:
    prefixes = {c.url_prefix for c in hrrr_fast_virtual_chunk_containers()}
    assert prefixes == {CACHE_LOCATION_PREFIX, S3_LOCATION_PREFIX}


@pytest.fixture
def listings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records the location_prefix of each discovery listing, in call order."""
    order: list[str] = []

    def fake_listing(
        pending: list[CoordT],
        *,
        store: object,
        location_prefix: str,
        require_index: bool,
    ) -> list[tuple[CoordT, int]]:
        order.append(location_prefix)
        assert require_index, "an index is the cache's write-completion signal"
        # Only the cache pass finds anything, so the rest must fall through.
        if location_prefix == CACHE_LOCATION_PREFIX:
            return [(pending[0], 100)] if pending else []
        return [(coord, 200) for coord in pending]

    monkeypatch.setattr(module, "discover_available_by_obstore_listing", fake_listing)
    monkeypatch.setattr(module, "s3_store", lambda *a, **k: object())
    return order


def test_discovery_tries_cache_before_archive(listings: list[str]) -> None:
    job = NoaaHrrrForecast48HourVirtualFastRegionJob.__new__(
        NoaaHrrrForecast48HourVirtualFastRegionJob
    )
    coords = [make_coord(), make_coord()]
    found = job.discover_available(coords)

    assert listings == [CACHE_LOCATION_PREFIX, S3_LOCATION_PREFIX]
    assert len(found) == 2
    # The cache hit keeps its low-latency source; the miss falls back.
    assert coords[0].source == "cache"
    assert coords[1].source == "archive"


def test_discovery_returns_the_same_coord_objects(listings: list[str]) -> None:
    """The write loop drops coords by identity, so copies would never be dropped."""
    job = NoaaHrrrForecast48HourVirtualFastRegionJob.__new__(
        NoaaHrrrForecast48HourVirtualFastRegionJob
    )
    coords = [make_coord(), make_coord()]
    found = job.discover_available(coords)
    assert {id(coord) for coord, _size in found} == {id(coord) for coord in coords}
