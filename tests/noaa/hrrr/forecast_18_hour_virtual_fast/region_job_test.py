from collections.abc import Callable, Iterator, Sequence
from itertools import count
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import obstore.store
import pandas as pd
import pydantic
import pytest

from reformatters.common import template_utils
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.noaa import noaa_virtual_region_job as shared_region_job_module
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.dynamical_dataset import (
    NoaaHrrrForecast18HourVirtualFastDataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.region_job import (
    PENDING_REPOINT_JOB_NAME,
    NoaaHrrrForecast18HourVirtualFastRegionJob,
    NoaaHrrrForecast18HourVirtualFastSourceFileCoord,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.template_config import (
    NoaaHrrrForecast18HourVirtualFastTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.nomads_mirror import MIRROR_LOCATION_PREFIX, mirror_key
from reformatters.noaa.hrrr.virtual_region_job import (
    S3_LOCATION_PREFIX,
    NoaaHrrrForecastVirtualRegionJob,
    NoaaHrrrForecastVirtualSourceFileCoord,
    NoaaHrrrVirtualRegionJob,
)

TEMPLATE_CONFIG = NoaaHrrrForecast18HourVirtualFastTemplateConfig()
FIXTURES = Path(__file__).parents[2] / "fixtures"
FIXTURE_GRIB = FIXTURES / "hrrr.t19z.wrfsfcf00.first2.grib2"
FIXTURE_INDEX = FIXTURES / "hrrr.t19z.wrfsfcf00.first2.grib2.idx"
MIRRORED_INIT = pd.Timestamp("2026-09-07T19:00")
LEAD_0 = pd.Timedelta("0h")
_copy_counter = count()


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


def test_operational_update_jobs_cover_six_hourly_cycles_and_repoint_mirrored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-09-02T01:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now))

    jobs, template_ds = (
        NoaaHrrrForecast18HourVirtualFastRegionJob.operational_update_jobs(
            primary_store=Mock(),
            tmp_store=Path("unused-tmp.zarr"),
            get_template_fn=TEMPLATE_CONFIG.get_template,
            append_dim="init_time",
            all_data_vars=TEMPLATE_CONFIG.data_vars,
            reformat_job_name="test",
        )
    )

    (job,) = jobs
    assert isinstance(job, NoaaHrrrForecast18HourVirtualFastRegionJob)
    assert job.processing_mode == "update"
    assert job.repoint_mirrored
    init_times = template_ds.to_dataset().get_index("init_time")
    assert len(init_times) > 6
    assert job.region == slice(len(init_times) - 6, len(init_times))


# --- Routing between NODD and the mirror ---


def get_var(name: str) -> NoaaHrrrDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.name == name)


def routed_coord(
    data_vars: Sequence[NoaaHrrrDataVar] | None = None,
    lead_time: pd.Timedelta = LEAD_0,
    init_time: pd.Timestamp = MIRRORED_INIT,
) -> NoaaHrrrForecast18HourVirtualFastSourceFileCoord:
    return NoaaHrrrForecast18HourVirtualFastSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        domain="conus",
        file_type="sfc",
        data_vars=data_vars or [get_var("composite_reflectivity")],
    )


def routed(
    coord: NoaaHrrrForecastVirtualSourceFileCoord,
) -> NoaaHrrrForecast18HourVirtualFastSourceFileCoord:
    assert isinstance(coord, NoaaHrrrForecast18HourVirtualFastSourceFileCoord)
    return coord


def make_dataset(tmp_path: Path) -> NoaaHrrrForecast18HourVirtualFastDataset:
    return NoaaHrrrForecast18HourVirtualFastDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "store"), format=DatasetFormat.ICECHUNK
        )
    )


def local_mirror_job_class(
    tmp_path: Path,
) -> type[NoaaHrrrForecast18HourVirtualFastRegionJob]:
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir(exist_ok=True)

    class LocalMirrorJob(NoaaHrrrForecast18HourVirtualFastRegionJob):
        def mirror_store(self) -> obstore.store.ObjectStore:
            return obstore.store.LocalStore(mirror_dir)

    return LocalMirrorJob


def make_job(
    tmp_path: Path,
    data_vars: Sequence[NoaaHrrrDataVar],
    processing_mode: Literal["backfill", "update"] = "update",
) -> NoaaHrrrForecast18HourVirtualFastRegionJob:
    template_ds = TEMPLATE_CONFIG.get_template(MIRRORED_INIT + pd.Timedelta("1h"))
    job = local_mirror_job_class(tmp_path)(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode=processing_mode,
        repoint_mirrored=True,
    )
    job.bind_store_factory(make_dataset(tmp_path).store_factory)
    return job


def record_name(coord: NoaaHrrrForecastVirtualSourceFileCoord) -> str:
    return mirror_key(coord).replace("/", "__")


def pending_records(store_factory: StoreFactory) -> list[str]:
    return store_factory.list_coordination_files(PENDING_REPOINT_JOB_NAME, "")


def write_record(
    store_factory: StoreFactory, coord: NoaaHrrrForecastVirtualSourceFileCoord
) -> None:
    store_factory.write_coordination_file(
        PENDING_REPOINT_JOB_NAME, record_name(coord), b""
    )


def mirror_file(tmp_path: Path, coord: NoaaHrrrForecastVirtualSourceFileCoord) -> None:
    path = tmp_path / "mirror" / mirror_key(coord)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FIXTURE_GRIB.read_bytes())
    path.with_name(path.name + ".idx").write_bytes(FIXTURE_INDEX.read_bytes())


def _nodd_listing(
    available: set[tuple[pd.Timestamp, pd.Timedelta, str]],
    probed: list[list[pd.Timedelta]] | None = None,
) -> Callable[..., list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]]:
    def discover(
        self: NoaaHrrrForecastVirtualRegionJob,
        pending: list[NoaaHrrrForecastVirtualSourceFileCoord],
    ) -> list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]:
        if probed is not None:
            probed.append([c.lead_time for c in pending])
        return [
            (coord, 16)
            for coord in pending
            if (coord.init_time, coord.lead_time, coord.file_type) in available
        ]

    return discover


def test_routing_fields_are_assignable_and_identity_is_frozen() -> None:
    coord = routed_coord()
    assert coord.bucket == "nodd"
    assert coord.get_url() == S3_LOCATION_PREFIX + mirror_key(coord)
    coord.route_to("mirror")
    assert coord.get_url() == MIRROR_LOCATION_PREFIX + mirror_key(coord)
    assert coord.get_index_url() == MIRROR_LOCATION_PREFIX + mirror_key(coord) + ".idx"
    coord.mark_present()
    assert coord.already_present
    with pytest.raises(pydantic.ValidationError):
        coord.lead_time = pd.Timedelta("1h")  # ty: ignore[invalid-assignment]


def test_index_and_data_reads_follow_the_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    coords = job.generate_source_file_coords(job._processing_region_ds(), job.data_vars)
    assert coords
    coord = routed(coords[0])
    mirror_file(tmp_path, coord)
    nodd_file(tmp_path, coord, grib=b"NODD-GRIB-BYTES", index=b"NODD-INDEX")
    nodd_reads = stub_nodd_reads(monkeypatch, tmp_path)

    index = job.download_index(coord)
    assert index.read_bytes() == b"NODD-INDEX"
    assert job.read_data_bytes(coord, 0, 4) == b"NODD"
    assert nodd_reads == [
        S3_LOCATION_PREFIX + mirror_key(coord) + ".idx",
        S3_LOCATION_PREFIX + mirror_key(coord),
    ]

    coord.route_to("mirror")
    index = job.download_index(coord)
    assert index.read_bytes() == FIXTURE_INDEX.read_bytes()
    index.unlink()
    assert job.read_data_bytes(coord, 0, 16) == FIXTURE_GRIB.read_bytes()[:16]
    assert len(nodd_reads) == 2


def test_discover_prefers_nodd_then_the_mirror_then_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    both = routed_coord(lead_time=LEAD_0)
    mirror_only = routed_coord(lead_time=pd.Timedelta("1h"))
    nowhere = routed_coord(lead_time=pd.Timedelta("2h"))
    mirror_file(tmp_path, both)
    mirror_file(tmp_path, mirror_only)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "discover_available",
        _nodd_listing({(MIRRORED_INIT, LEAD_0, "sfc")}),
    )

    found = job.discover_available([both, mirror_only, nowhere])

    assert [id(c) for c, _ in found] == [id(both), id(mirror_only)]
    assert both.bucket == "nodd"
    assert mirror_only.bucket == "mirror"
    assert found[1][1] == FIXTURE_GRIB.stat().st_size
    assert nowhere.bucket == "nodd"


def test_a_present_file_is_only_ever_resupplied_by_nodd_and_probed_once_a_minute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    monkeypatch.setattr(type(job), "repoint_probe_every", 3)
    present = routed_coord(lead_time=LEAD_0)
    present.mark_present()
    mirror_file(tmp_path, present)
    fresh = routed_coord(lead_time=pd.Timedelta("1h"))
    probed: list[list[pd.Timedelta]] = []
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob, "discover_available", _nodd_listing(set(), probed)
    )

    for _ in range(4):
        assert job.discover_available([present, fresh]) == []
    # The present file is in the mirror but never re-read from it, and NODD is asked
    # about it on ticks 1 and 4 only; the fresh file is asked about every tick.
    assert probed == [
        [LEAD_0, pd.Timedelta("1h")],
        [pd.Timedelta("1h")],
        [pd.Timedelta("1h")],
        [LEAD_0, pd.Timedelta("1h")],
    ]
    assert present.bucket == "nodd"

    monkeypatch.setattr(type(job), "repoint_probe_every", 1)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "discover_available",
        _nodd_listing({(MIRRORED_INIT, LEAD_0, "sfc")}),
    )
    assert job.discover_available([present]) == [(present, 16)]


def test_discover_is_nodd_only_for_backfills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(
        tmp_path, [get_var("composite_reflectivity")], processing_mode="backfill"
    )
    coord = routed_coord()
    mirror_file(tmp_path, coord)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob, "discover_available", _nodd_listing(set())
    )
    assert job.discover_available([coord]) == []
    assert coord.bucket == "nodd"


def test_an_unreachable_mirror_leaves_nodd_as_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    coord = routed_coord()

    def broken_store(
        self: NoaaHrrrForecast18HourVirtualFastRegionJob,
    ) -> obstore.store.ObjectStore:
        raise RuntimeError("mirror bucket unreachable")

    monkeypatch.setattr(type(job), "mirror_store", broken_store)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "discover_available",
        _nodd_listing({(MIRRORED_INIT, LEAD_0, "sfc")}),
    )
    assert job.discover_available([coord]) == [(coord, 16)]
    assert coord.bucket == "nodd"


def test_filter_re_offers_a_present_file_with_a_pending_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    absent = routed_coord(lead_time=LEAD_0)
    present_pending = routed_coord(lead_time=pd.Timedelta("1h"))
    present_final = routed_coord(lead_time=pd.Timedelta("2h"))
    write_record(job._bound_store_factory(), present_pending)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "filter_already_present",
        lambda self, candidates, store: [absent],
    )

    remaining = job.filter_already_present(
        [absent, present_pending, present_final], Mock()
    )

    assert [id(c) for c in remaining] == [id(absent), id(present_pending)]
    assert not absent.already_present
    assert present_pending.already_present

    # The validation job asks the plain question.
    plain = job.model_copy(update={"repoint_mirrored": False})
    assert plain.filter_already_present(
        [absent, present_pending, present_final], Mock()
    ) == [absent]


def test_filter_ingests_fresh_a_file_whose_record_has_no_refs_in_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record survives a commit that failed after it was written; the file is then
    absent from the store and offered as a fresh candidate, not as a repoint."""
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    absent = routed_coord()
    write_record(job._bound_store_factory(), absent)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "filter_already_present",
        lambda self, candidates, store: [absent],
    )

    remaining = job.filter_already_present([absent], Mock())

    assert [id(c) for c in remaining] == [id(absent)]
    assert not absent.already_present


def test_filter_recovers_a_pending_file_from_outside_the_update_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file mirrored two days ago whose refs still point at the mirror is re-offered
    even though the update window has moved on; one outside the template is not."""
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    old = routed_coord(init_time=MIRRORED_INIT - pd.Timedelta("48h"))
    beyond_template = routed_coord(lead_time=pd.Timedelta("40h"))
    write_record(job._bound_store_factory(), old)
    write_record(job._bound_store_factory(), beyond_template)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "filter_already_present",
        lambda self, candidates, store: [],
    )
    remaining = job.filter_already_present([], Mock())
    (recovered,) = (routed(c) for c in remaining)
    assert (recovered.init_time, recovered.lead_time, recovered.file_type) == (
        old.init_time,
        old.lead_time,
        "sfc",
    )
    assert recovered.already_present
    assert [v.name for v in recovered.data_vars] == ["composite_reflectivity"]


def test_filter_offers_fresh_a_recorded_file_outside_the_window_with_no_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record left by a failed commit, for a file the update window has moved past,
    is a fresh ingest again (mirror allowed), not a NODD-only repoint of nothing."""
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    old = routed_coord(init_time=MIRRORED_INIT - pd.Timedelta("48h"))
    write_record(job._bound_store_factory(), old)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "filter_already_present",
        lambda self, candidates, store: list(candidates),
    )
    (recovered,) = (routed(c) for c in job.filter_already_present([], Mock()))
    assert (recovered.init_time, recovered.lead_time) == (old.init_time, LEAD_0)
    assert not recovered.already_present
    assert pending_records(job._bound_store_factory()) == [record_name(old)]


def test_the_cap_keeps_slots_for_the_newest_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records NODD never satisfies pile up at the old end; a file recorded after
    them must still be re-offered."""
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    monkeypatch.setattr(type(job), "max_repoints_per_fire", 4)
    dead = [
        routed_coord(init_time=MIRRORED_INIT - pd.Timedelta(hours=h))
        for h in range(30, 24, -1)
    ]
    new = routed_coord(init_time=MIRRORED_INIT - pd.Timedelta("1h"))
    for coord in [*dead, new]:
        write_record(job._bound_store_factory(), coord)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "filter_already_present",
        lambda self, candidates, store: [],
    )
    offered = {routed(c).init_time for c in job.filter_already_present([], Mock())}
    assert new.init_time in offered
    assert dead[0].init_time in offered
    assert len(offered) == 4


def test_a_file_missing_an_expected_message_is_skipped_not_ingested_in_part(
    tmp_path: Path,
) -> None:
    """The fixture holds the first two sfc messages, so temperature_2m has none: the
    file is skipped whole rather than committed with that variable unfilled."""
    job = make_job(
        tmp_path, [get_var("composite_reflectivity"), get_var("temperature_2m")]
    )
    coord = routed_coord([get_var("composite_reflectivity"), get_var("temperature_2m")])
    mirror_file(tmp_path, coord)
    coord.route_to("mirror")
    size = (tmp_path / "mirror" / mirror_key(coord)).stat().st_size
    with pytest.raises(ValueError, match="temperature_2m"):
        job.file_refs(coord, size)
    assert job._file_refs_or_skip(coord, size) == []

    whole = routed_coord([get_var("composite_reflectivity")])
    whole.route_to("mirror")
    assert len(job.file_refs(whole, size)) == 1


def test_filter_re_offers_the_oldest_and_newest_pending_records_each_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    monkeypatch.setattr(type(job), "max_repoints_per_fire", 2)
    coords = [
        routed_coord(init_time=MIRRORED_INIT - pd.Timedelta(hours=hours))
        for hours in (1, 30, 5)
    ]
    for coord in coords:
        write_record(job._bound_store_factory(), coord)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "filter_already_present",
        lambda self, candidates, store: [],
    )

    remaining = job.filter_already_present([], Mock())

    assert [c.init_time for c in remaining] == [
        MIRRORED_INIT - pd.Timedelta("30h"),
        MIRRORED_INIT - pd.Timedelta("1h"),
    ]


def test_one_record_per_file_is_written_before_the_commit_and_deleted_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One record per file, however many coords name the file in a batch."""
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    store_factory = job._bound_store_factory()
    first = routed_coord([get_var("composite_reflectivity")])
    second = routed_coord([get_var("temperature_2m")])
    for coord in (first, second):
        coord.route_to("mirror")
    batches: list[
        Sequence[tuple[NoaaHrrrForecastVirtualSourceFileCoord, Sequence[VirtualRef]]]
    ] = [[(first, []), (second, [])]]

    def base_batches(
        self: NoaaHrrrForecastVirtualRegionJob,
        remaining: Sequence[NoaaHrrrForecastVirtualSourceFileCoord],
    ) -> Iterator[
        Sequence[tuple[NoaaHrrrForecastVirtualSourceFileCoord, Sequence[VirtualRef]]]
    ]:
        yield from batches

    monkeypatch.setattr(NoaaHrrrVirtualRegionJob, "process_virtual_refs", base_batches)
    writes: list[str] = []
    deletes: list[str] = []
    original_write = StoreFactory.write_coordination_file
    original_delete = StoreFactory.delete_coordination_file

    def counting_write(
        self: StoreFactory, job_name: str, key: str, data: bytes
    ) -> None:
        writes.append(key)
        original_write(self, job_name, key, data)

    def counting_delete(self: StoreFactory, job_name: str, key: str) -> None:
        deletes.append(key)
        original_delete(self, job_name, key)

    monkeypatch.setattr(StoreFactory, "write_coordination_file", counting_write)
    monkeypatch.setattr(StoreFactory, "delete_coordination_file", counting_delete)

    refs = job.process_virtual_refs([first, second])
    assert next(refs) == batches[0]
    # Yielded but not yet committed: the record already exists.
    assert pending_records(store_factory) == [record_name(first)]
    assert list(refs) == []
    assert writes == [record_name(first)]
    assert deletes == []

    for coord in (first, second):
        coord.route_to("nodd")
        coord.mark_present()
    refs = job.process_virtual_refs([first, second])
    assert next(refs) == batches[0]
    # Not deleted until the repointed refs are committed.
    assert pending_records(store_factory) == [record_name(first)]
    assert list(refs) == []
    assert deletes == [record_name(first)]
    assert pending_records(store_factory) == []


def nodd_file(
    tmp_path: Path,
    coord: NoaaHrrrForecastVirtualSourceFileCoord,
    grib: bytes | None = None,
    index: bytes | None = None,
) -> None:
    path = tmp_path / "nodd" / mirror_key(coord)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FIXTURE_GRIB.read_bytes() if grib is None else grib)
    path.with_name(path.name + ".idx").write_bytes(
        FIXTURE_INDEX.read_bytes() if index is None else index
    )


def stub_nodd_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Serve the NODD index downloads and GRIB byte reads from `tmp_path/nodd`;
    returns the list of URLs read, appended to as reads happen."""
    urls: list[str] = []

    def local_path(url: str) -> Path:
        assert url.startswith(S3_LOCATION_PREFIX), url
        urls.append(url)
        return tmp_path / "nodd" / url.removeprefix(S3_LOCATION_PREFIX)

    def download(url: str, dataset_id: str, **kwargs: object) -> Path:
        copy = tmp_path / f"{next(_copy_counter)}.idx"
        copy.write_bytes(local_path(url).read_bytes())
        return copy

    def read_bytes(url: str, *, start: int, end: int, **kwargs: object) -> bytes:
        return local_path(url).read_bytes()[start:end]

    monkeypatch.setattr(shared_region_job_module, "s3_download_to_disk", download)
    monkeypatch.setattr(shared_region_job_module, "s3_read_bytes", read_bytes)
    return urls


class LocalRepo:
    """A local icechunk store for the fast dataset, trimmed to lead 0, with NODD
    listings and reads served from `tmp_path/nodd`."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.dataset = make_dataset(tmp_path)
        template_config = self.dataset.template_config
        original_get_template = template_config.get_template
        monkeypatch.setattr(
            type(template_config),
            "get_template",
            lambda self, end_time: original_get_template(end_time).isel(lead_time=[0]),
        )
        self.template_ds = template_config.get_template(
            MIRRORED_INIT + pd.Timedelta("1h")
        )
        template_utils.write_metadata(self.template_ds, self.store_factory)
        (_, self.repo), *_ = self.store_factory.icechunk_repos(sort="primary-first")
        self.position = int(
            self.template_ds.to_dataset()
            .get_index("init_time")
            .get_indexer(pd.Index([MIRRORED_INIT]))[0]
        )
        nodd_dir = tmp_path / "nodd"
        nodd_dir.mkdir()
        monkeypatch.setattr(
            shared_region_job_module,
            "s3_store",
            lambda bucket_url, region, **kwargs: obstore.store.LocalStore(nodd_dir),
        )
        stub_nodd_reads(monkeypatch, tmp_path)
        self.job_class = local_mirror_job_class(tmp_path)
        monkeypatch.setattr(self.job_class, "tick_interval", pd.Timedelta("0s"))

    @property
    def store_factory(self) -> StoreFactory:
        return self.dataset.store_factory

    def fire(self, data_vars: Sequence[NoaaHrrrDataVar]) -> None:
        job = self.job_class(
            tmp_store=Path("unused-tmp.zarr"),
            template_ds=self.template_ds,
            data_vars=data_vars,
            append_dim="init_time",
            region=slice(self.position, self.position + 1),
            reformat_job_name="test",
            processing_mode="update",
            repoint_mirrored=True,
            poll_deadline=pd.Timestamp.now() + pd.Timedelta("5s"),
        )
        job.bind_store_factory(self.store_factory)
        remaining = job.filter_already_present(
            job.source_file_coords(), self.repo.readonly_session("main").store
        )
        job.process_virtual(self.repo, [], "main", remaining)

    def locations(self) -> list[str]:
        return self.repo.readonly_session("main").all_virtual_chunk_locations()


def test_a_mirror_file_is_ingested_then_repointed_to_nodd_on_the_next_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end against a local icechunk store: fire 1 ingests the file from the
    mirror the tick it appears there and records it; fire 2 rewrites its refs from
    NODD and drops the record, though the mirror has expired the file by then."""
    local = LocalRepo(tmp_path, monkeypatch)
    data_vars = [get_var("composite_reflectivity")]
    coord = routed_coord(data_vars)
    mirror_file(tmp_path, coord)
    mirror_url = MIRROR_LOCATION_PREFIX + mirror_key(coord)
    nodd_url = S3_LOCATION_PREFIX + mirror_key(coord)

    local.fire(data_vars)
    assert mirror_url in local.locations()
    assert nodd_url not in local.locations()
    assert pending_records(local.store_factory) == [record_name(coord)]

    for path in (tmp_path / "mirror").rglob("*.grib2*"):
        path.unlink()
    nodd_file(tmp_path, coord)
    local.fire(data_vars)
    assert nodd_url in local.locations()
    assert mirror_url not in local.locations()
    assert pending_records(local.store_factory) == []


def test_a_record_left_by_a_failed_commit_ingests_the_file_fresh_from_nodd_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = LocalRepo(tmp_path, monkeypatch)
    data_vars = [get_var("composite_reflectivity")]
    coord = routed_coord(data_vars)
    write_record(local.store_factory, coord)
    mirror_file(tmp_path, coord)
    nodd_file(tmp_path, coord)

    local.fire(data_vars)

    assert S3_LOCATION_PREFIX + mirror_key(coord) in local.locations()
    assert MIRROR_LOCATION_PREFIX + mirror_key(coord) not in local.locations()
    # The stale record re-offers the file to NODD next fire, which then drops it.
    assert pending_records(local.store_factory) == [record_name(coord)]
    local.fire(data_vars)
    assert pending_records(local.store_factory) == []
    assert S3_LOCATION_PREFIX + mirror_key(coord) in local.locations()
