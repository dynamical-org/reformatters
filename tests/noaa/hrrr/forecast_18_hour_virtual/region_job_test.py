from collections.abc import Callable, Sequence
from itertools import count
from pathlib import Path
from typing import ClassVar, Literal
from unittest.mock import Mock

import obstore.store
import pandas as pd
import pydantic
import pytest

from reformatters.common import template_utils
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.noaa import noaa_virtual_region_job as shared_region_job_module
from reformatters.noaa.hrrr.forecast_18_hour_virtual import (
    region_job as region_job_module,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast18HourVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual.region_job import (
    NoaaHrrrForecast18HourVirtualRegionJob,
    NoaaHrrrForecast18HourVirtualSourceFileCoord,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual.template_config import (
    NoaaHrrrForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.nomads_cache import (
    NOMADS_CACHE_LOCATION_PREFIX,
    REPOINTED_MARKER_SUFFIX,
    cache_key,
)
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord
from reformatters.noaa.hrrr.virtual_region_job import (
    S3_LOCATION_PREFIX,
    NoaaHrrrForecastVirtualRegionJob,
    NoaaHrrrForecastVirtualSourceFileCoord,
    NoaaHrrrVirtualRegionJob,
)

TEMPLATE_CONFIG = NoaaHrrrForecast18HourVirtualTemplateConfig()


def test_source_file_coord_url_non_synoptic_init() -> None:
    coord = NoaaHrrrForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T01:00"),
        lead_time=pd.Timedelta("18h"),
        domain="conus",
        file_type="sfc",
        data_vars=[TEMPLATE_CONFIG.data_vars[0]],
    )
    assert coord.get_url() == (
        "s3://noaa-hrrr-bdp-pds/hrrr.20240601/conus/hrrr.t01z.wrfsfcf18.grib2"
    )


def test_operational_update_jobs_cover_six_hourly_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2024-06-02T01:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now))

    jobs, template_ds = NoaaHrrrForecast18HourVirtualRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="init_time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )

    (job,) = jobs
    assert isinstance(job, NoaaHrrrForecast18HourVirtualRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    assert job.region == slice(len(init_times) - 6, len(init_times))


def test_generate_source_file_coords_uses_the_declared_coord_class() -> None:
    class RoutedCoord(NoaaHrrrForecastVirtualSourceFileCoord):
        pass

    class RoutedJob(NoaaHrrrForecast18HourVirtualRegionJob):
        source_file_coord_class = RoutedCoord

    template_ds = TEMPLATE_CONFIG.get_template(pd.Timestamp("2018-07-13T13:00"))
    job = RoutedJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=TEMPLATE_CONFIG.data_vars[:2],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
    coords = job.generate_source_file_coords(job._processing_region_ds(), job.data_vars)
    assert coords
    assert all(type(coord) is RoutedCoord for coord in coords)


# --- Cache-first routing ---

FIXTURES = Path(__file__).parents[2] / "fixtures"
FIXTURE_INDEX = FIXTURES / "hrrr.t19z.wrfsfcf00.first2.grib2.idx"
FIXTURE_GRIB = FIXTURES / "hrrr.t19z.wrfsfcf00.first2.grib2"
CACHED_INIT = pd.Timestamp("2026-09-07T19:00")


def get_var(name: str) -> NoaaHrrrDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.name == name)


LEAD_0 = pd.Timedelta("0h")


def routed_coord(
    data_vars: Sequence[NoaaHrrrDataVar] | None = None,
    lead_time: pd.Timedelta = LEAD_0,
    init_time: pd.Timestamp = CACHED_INIT,
) -> NoaaHrrrForecast18HourVirtualSourceFileCoord:
    return NoaaHrrrForecast18HourVirtualSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        domain="conus",
        file_type="sfc",
        data_vars=data_vars or [get_var("composite_reflectivity"), get_var("echo_top")],
    )


def make_job(
    tmp_path: Path,
    data_vars: Sequence[NoaaHrrrDataVar],
    processing_mode: Literal["backfill", "update"] = "update",
) -> NoaaHrrrForecast18HourVirtualRegionJob:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)

    class LocalCacheJob(NoaaHrrrForecast18HourVirtualRegionJob):
        def cache_store(self) -> obstore.store.ObjectStore:
            return obstore.store.LocalStore(cache_dir)

        def cache_writer(self) -> obstore.store.ObjectStore:
            return obstore.store.LocalStore(cache_dir)

    template_ds = TEMPLATE_CONFIG.get_template(CACHED_INIT + pd.Timedelta("1h"))
    return LocalCacheJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode=processing_mode,
    )


def cache_file(tmp_path: Path, coord: NoaaHrrrSourceFileCoord, index: str) -> None:
    path = tmp_path / "cache" / cache_key(coord)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GRIB" * 4)
    path.with_name(path.name + ".idx").write_text(index)


def test_routing_fields_are_assignable_and_identity_is_frozen() -> None:
    coord = routed_coord()
    assert coord.bucket == "nodd"
    assert coord.get_url() == (
        "s3://noaa-hrrr-bdp-pds/hrrr.20260907/conus/hrrr.t19z.wrfsfcf00.grib2"
    )
    coord.route_to("cache")
    assert coord.get_url() == (
        "s3://dynamical-noaa-hrrr-nomads/hrrr.20260907/conus/hrrr.t19z.wrfsfcf00.grib2"
    )
    assert coord.get_index_url().endswith(".grib2.idx")
    coord.reject_cache()
    assert coord.cache_rejected
    with pytest.raises(pydantic.ValidationError):
        coord.lead_time = pd.Timedelta("1h")  # ty: ignore[invalid-assignment]
    twin = coord.repoint_twin()
    assert twin.repoint
    assert twin.bucket == "nodd"
    assert not twin.cache_rejected
    assert twin.file_key() == coord.file_key()
    assert twin.data_vars[0] is coord.data_vars[0]


def test_source_region_follows_the_bucket(tmp_path: Path) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    coord = routed_coord()
    assert job.source_region(coord) == "us-east-1"
    coord.route_to("cache")
    assert job.source_region(coord) == "us-west-2"


def test_generate_source_file_coords_are_routable(tmp_path: Path) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    coords = job.generate_source_file_coords(job._processing_region_ds(), job.data_vars)
    assert all(
        isinstance(c, NoaaHrrrForecast18HourVirtualSourceFileCoord) for c in coords
    )


def _nodd_listing(
    available: set[tuple[pd.Timestamp, pd.Timedelta, str]],
) -> Callable[..., list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]]:
    def discover(
        self: NoaaHrrrForecastVirtualRegionJob,
        pending: list[NoaaHrrrForecastVirtualSourceFileCoord],
    ) -> list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]:
        return [
            (coord, 16)
            for coord in pending
            if (coord.init_time, coord.lead_time, coord.file_type) in available
        ]

    return discover


def test_discover_available_prefers_the_cache_then_nodd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity"), get_var("echo_top")])
    cached = routed_coord(lead_time=pd.Timedelta("0h"))
    on_nodd = routed_coord(lead_time=pd.Timedelta("1h"))
    nowhere = routed_coord(lead_time=pd.Timedelta("2h"))
    cache_file(tmp_path, cached, FIXTURE_INDEX.read_text())
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "discover_available",
        _nodd_listing({(CACHED_INIT, pd.Timedelta("1h"), "sfc")}),
    )
    stub_cache_index_reads(monkeypatch, tmp_path)

    found = job.discover_available([cached, on_nodd, nowhere])

    assert found == [(cached, 16), (on_nodd, 16)]
    assert found[0][0] is cached
    assert cached.bucket == "cache"
    assert found[1][0] is on_nodd
    assert on_nodd.bucket == "nodd"
    assert nowhere.bucket == "nodd"
    assert not nowhere.cache_rejected


def test_discover_available_rejects_a_cache_index_missing_a_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cached f00 index only names REFC and RETOP; the coord also needs TMP.
    job = make_job(
        tmp_path, [get_var("composite_reflectivity"), get_var("temperature_2m")]
    )
    coord = routed_coord([get_var("composite_reflectivity"), get_var("temperature_2m")])
    cache_file(tmp_path, coord, FIXTURE_INDEX.read_text())
    nodd_has_it = {(CACHED_INIT, pd.Timedelta("0h"), "sfc")}
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob, "discover_available", _nodd_listing(set())
    )
    stub_cache_index_reads(monkeypatch, tmp_path)

    assert job.discover_available([coord]) == []
    assert coord.cache_rejected
    assert coord.bucket == "nodd"

    # Next tick NODD has the file: the rejected coord goes to NODD, never the cache.
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob, "discover_available", _nodd_listing(nodd_has_it)
    )
    assert job.discover_available([coord]) == [(coord, 16)]
    assert coord.bucket == "nodd"


def test_discover_available_probes_repoint_twins_on_nodd_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    twin = routed_coord([get_var("composite_reflectivity")]).repoint_twin()
    cache_file(tmp_path, twin, FIXTURE_INDEX.read_text())
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob, "discover_available", _nodd_listing(set())
    )
    stub_cache_index_reads(monkeypatch, tmp_path)
    assert job.discover_available([twin]) == []
    assert twin.bucket == "nodd"


def test_discover_available_is_nodd_only_for_backfills_and_when_cache_first_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord = routed_coord([get_var("composite_reflectivity")])
    cache_file(tmp_path, coord, FIXTURE_INDEX.read_text())
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob, "discover_available", _nodd_listing(set())
    )
    backfill = make_job(tmp_path, [get_var("composite_reflectivity")], "backfill")
    assert backfill.discover_available([coord]) == []
    update = make_job(tmp_path, [get_var("composite_reflectivity")])
    monkeypatch.setattr(type(update), "cache_first", False)
    assert update.discover_available([coord]) == []
    assert coord.bucket == "nodd"


def test_committed_schedules_a_repoint_twin_and_marks_a_finished_repoint(
    tmp_path: Path,
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    from_cache = routed_coord([get_var("composite_reflectivity")])
    from_cache.route_to("cache")
    from_nodd = routed_coord(
        [get_var("composite_reflectivity")], lead_time=pd.Timedelta("1h")
    )
    twin = routed_coord(
        [get_var("composite_reflectivity")], lead_time=pd.Timedelta("2h")
    ).repoint_twin()
    cache_file(tmp_path, twin, FIXTURE_INDEX.read_text())

    follow_ups = job.committed([(from_cache, []), (from_nodd, []), (twin, [])])

    (follow_up,) = (routed(c) for c in follow_ups)
    assert follow_up.repoint
    assert follow_up.file_key() == from_cache.file_key()
    marker = tmp_path / "cache" / (cache_key(twin) + REPOINTED_MARKER_SUFFIX)
    assert marker.exists()
    assert not (
        tmp_path / "cache" / (cache_key(from_cache) + REPOINTED_MARKER_SUFFIX)
    ).exists()


def test_check_refs_complete_guards_cache_files_and_repoints_only(
    tmp_path: Path,
) -> None:
    data_vars = [get_var("composite_reflectivity"), get_var("echo_top")]
    job = make_job(tmp_path, data_vars)
    coord = routed_coord(data_vars)
    partial = [
        VirtualRef(
            data_var=get_var("composite_reflectivity"),
            out_loc=coord.out_loc(),
            location=coord.get_url(),
            offset=0,
            length=16,
        )
    ]
    job._check_refs_complete(coord, partial)  # plain NODD: partial files are allowed
    coord.route_to("cache")
    with pytest.raises(ValueError, match="echo_top"):
        job._check_refs_complete(coord, partial)
    with pytest.raises(ValueError, match="echo_top"):
        job._check_refs_complete(coord.repoint_twin(), partial)


def test_unfinished_work_merges_the_window_with_the_cache_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_vars = [get_var("composite_reflectivity")]
    job = make_job(tmp_path, data_vars)
    window = [routed(c) for c in job.source_file_coords()]
    present_cached, present_plain, absent = window[0], window[1], window[2]
    cache_file(tmp_path, present_cached, FIXTURE_INDEX.read_text())
    cache_file(tmp_path, absent, FIXTURE_INDEX.read_text())
    # A cached file from an init outside the window, already ingested from the cache.
    older = routed_coord(data_vars, init_time=CACHED_INIT - pd.Timedelta("8h"))
    cache_file(tmp_path, older, FIXTURE_INDEX.read_text())
    # And one already repointed: nothing left to do for it.
    done = routed_coord(data_vars, init_time=CACHED_INIT - pd.Timedelta("9h"))
    cache_file(tmp_path, done, FIXTURE_INDEX.read_text())
    (tmp_path / "cache" / (cache_key(done) + REPOINTED_MARKER_SUFFIX)).write_bytes(b"")

    absent_keys = {absent.file_key()}
    monkeypatch.setattr(
        NoaaHrrrForecast18HourVirtualRegionJob,
        "filter_already_present",
        lambda self, candidates, store: [
            c for c in candidates if routed(c).file_key() in absent_keys
        ],
    )

    work = [routed(c) for c in job.unfinished_work(Mock())]

    by_key = {(c.file_key(), c.repoint) for c in work}
    assert by_key == {
        (absent.file_key(), False),
        (present_cached.file_key(), True),
        (older.file_key(), True),
    }
    assert present_plain.file_key() not in {c.file_key() for c in work}
    assert len(work) == 3


def routed(
    coord: NoaaHrrrForecastVirtualSourceFileCoord,
) -> NoaaHrrrForecast18HourVirtualSourceFileCoord:
    assert isinstance(coord, NoaaHrrrForecast18HourVirtualSourceFileCoord)
    return coord


def stub_cache_index_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Serve `s3_download_to_disk` for cache index URLs from the local cache dir."""

    def download(url: str, dataset_id: str, **kwargs: object) -> Path:
        source = tmp_path / "cache" / url.removeprefix(NOMADS_CACHE_LOCATION_PREFIX)
        copy = tmp_path / f"{next(_copy_counter)}.idx"
        copy.write_bytes(source.read_bytes())
        return copy

    monkeypatch.setattr(region_job_module, "s3_download_to_disk", download)


_copy_counter = count()


def test_a_cache_file_is_repointed_to_nodd_within_one_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end against a local icechunk store: the file is ingested from the cache
    the tick it appears there, its refs are rewritten from NODD the tick NODD has it,
    and the cache object is then marked repointed."""
    cache_dir, nodd_dir = tmp_path / "cache", tmp_path / "nodd"
    dataset = NoaaHrrrForecast18HourVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "store"), format=DatasetFormat.ICECHUNK
        )
    )
    original_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: original_get_template(end_time).isel(lead_time=[0]),
    )
    template_ds = dataset.template_config.get_template(CACHED_INIT + pd.Timedelta("1h"))
    template_utils.write_metadata(template_ds, dataset.store_factory)
    (_, repo), *_ = dataset.store_factory.icechunk_repos(sort="primary-first")
    position = int(
        template_ds.to_dataset()
        .get_index("init_time")
        .get_indexer(pd.Index([CACHED_INIT]))[0]
    )
    data_vars = [get_var("composite_reflectivity")]
    coord = routed_coord(data_vars)
    cache_file(tmp_path, coord, FIXTURE_INDEX.read_text())
    (cache_dir / cache_key(coord)).write_bytes(FIXTURE_GRIB.read_bytes())

    def publish_on_nodd() -> None:
        target = nodd_dir / cache_key(coord)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(FIXTURE_GRIB.read_bytes())
        target.with_name(target.name + ".idx").write_text(FIXTURE_INDEX.read_text())

    class TwoSourceJob(NoaaHrrrForecast18HourVirtualRegionJob):
        ticks: ClassVar[int] = 0

        def cache_store(self) -> obstore.store.ObjectStore:
            return obstore.store.LocalStore(cache_dir)

        def cache_writer(self) -> obstore.store.ObjectStore:
            return obstore.store.LocalStore(cache_dir)

        def discover_available(
            self, pending: list[NoaaHrrrForecastVirtualSourceFileCoord]
        ) -> list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]:
            TwoSourceJob.ticks += 1
            if TwoSourceJob.ticks == 2:
                publish_on_nodd()
            return super().discover_available(pending)

    def local_dir(url: str) -> Path:
        if url.startswith(NOMADS_CACHE_LOCATION_PREFIX):
            return cache_dir / url.removeprefix(NOMADS_CACHE_LOCATION_PREFIX)
        return nodd_dir / url.removeprefix(S3_LOCATION_PREFIX)

    def download(url: str, dataset_id: str, **kwargs: object) -> Path:
        copy = tmp_path / f"{next(_copy_counter)}.idx"
        copy.write_bytes(local_dir(url).read_bytes())
        return copy

    def read_bytes(url: str, *, start: int, end: int, **kwargs: object) -> bytes:
        return local_dir(url).read_bytes()[start:end]

    nodd_dir.mkdir()
    monkeypatch.setattr(
        shared_region_job_module,
        "s3_store",
        lambda bucket_url, region, **kwargs: obstore.store.LocalStore(nodd_dir),
    )
    monkeypatch.setattr(shared_region_job_module, "s3_download_to_disk", download)
    monkeypatch.setattr(region_job_module, "s3_download_to_disk", download)
    monkeypatch.setattr(shared_region_job_module, "s3_read_bytes", read_bytes)
    monkeypatch.setattr(TwoSourceJob, "tick_interval", pd.Timedelta("0s"))

    job = TwoSourceJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(position, position + 1),
        reformat_job_name="test",
        processing_mode="update",
        poll_deadline=pd.Timestamp.now() + pd.Timedelta("60s"),
    )
    snapshots_before = [s.id for s in repo.ancestry(branch="main")]
    remaining = job.unfinished_work(repo.readonly_session("main").store)
    assert [routed(c).file_key() for c in remaining] == [coord.file_key()]

    job.process_virtual(repo, [], "main", remaining)

    head, after_cache, *_ = [
        s.id for s in repo.ancestry(branch="main") if s.id not in snapshots_before
    ]
    cache_url = NOMADS_CACHE_LOCATION_PREFIX + cache_key(coord)
    nodd_url = S3_LOCATION_PREFIX + cache_key(coord)
    assert (
        cache_url
        in repo.readonly_session(snapshot_id=after_cache).all_virtual_chunk_locations()
    )
    head_locations = repo.readonly_session(
        snapshot_id=head
    ).all_virtual_chunk_locations()
    assert nodd_url in head_locations
    assert cache_url not in head_locations
    assert (cache_dir / (cache_key(coord) + REPOINTED_MARKER_SUFFIX)).exists()
    assert TwoSourceJob.ticks == 2
