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
from reformatters.noaa import noaa_virtual_region_job as shared_region_job_module
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
from reformatters.noaa.hrrr.nomads_mirror import MIRROR_LOCATION_PREFIX, mirror_key
from reformatters.noaa.hrrr.virtual_region_job import (
    S3_LOCATION_PREFIX,
    NoaaHrrrForecastVirtualRegionJob,
    NoaaHrrrForecastVirtualSourceFileCoord,
    NoaaHrrrVirtualRegionJob,
)

TEMPLATE_CONFIG = NoaaHrrrForecast18HourVirtualTemplateConfig()
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
    assert job.repoint_mirrored
    init_times = template_ds.to_dataset().get_index("init_time")
    assert job.region == slice(len(init_times) - 6, len(init_times))


# --- Routing between NODD and the mirror ---


def get_var(name: str) -> NoaaHrrrDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.name == name)


def routed_coord(
    data_vars: Sequence[NoaaHrrrDataVar] | None = None,
    lead_time: pd.Timedelta = LEAD_0,
    init_time: pd.Timestamp = MIRRORED_INIT,
) -> NoaaHrrrForecast18HourVirtualSourceFileCoord:
    return NoaaHrrrForecast18HourVirtualSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        domain="conus",
        file_type="sfc",
        data_vars=data_vars or [get_var("composite_reflectivity")],
    )


def routed(
    coord: NoaaHrrrForecastVirtualSourceFileCoord,
) -> NoaaHrrrForecast18HourVirtualSourceFileCoord:
    assert isinstance(coord, NoaaHrrrForecast18HourVirtualSourceFileCoord)
    return coord


def make_job(
    tmp_path: Path,
    data_vars: Sequence[NoaaHrrrDataVar],
    processing_mode: Literal["backfill", "update"] = "update",
) -> NoaaHrrrForecast18HourVirtualRegionJob:
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir(exist_ok=True)

    class LocalMirrorJob(NoaaHrrrForecast18HourVirtualRegionJob):
        def mirror_store(self) -> obstore.store.ObjectStore:
            return obstore.store.LocalStore(mirror_dir)

    template_ds = TEMPLATE_CONFIG.get_template(MIRRORED_INIT + pd.Timedelta("1h"))
    return LocalMirrorJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode=processing_mode,
        repoint_mirrored=True,
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


def test_generate_source_file_coords_are_routable_and_regions_follow_the_bucket(
    tmp_path: Path,
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    coords = job.generate_source_file_coords(job._processing_region_ds(), job.data_vars)
    assert coords
    coord = routed(coords[0])
    assert job.source_region(coord) == "us-east-1"
    coord.route_to("mirror")
    assert job.source_region(coord) == "us-west-2"


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
        self: NoaaHrrrForecast18HourVirtualRegionJob,
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


def test_filter_keeps_offering_present_files_the_mirror_still_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    absent = routed_coord(lead_time=LEAD_0)
    present_in_mirror = routed_coord(lead_time=pd.Timedelta("1h"))
    present_final = routed_coord(lead_time=pd.Timedelta("2h"))
    mirror_file(tmp_path, absent)
    mirror_file(tmp_path, present_in_mirror)
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "filter_already_present",
        lambda self, candidates, store: [absent],
    )

    remaining = job.filter_already_present(
        [absent, present_in_mirror, present_final], Mock()
    )

    assert [id(c) for c in remaining] == [id(absent), id(present_in_mirror)]
    assert not absent.already_present
    assert present_in_mirror.already_present

    # The validation job asks the plain question.
    plain = job.model_copy(update={"repoint_mirrored": False})
    assert plain.filter_already_present(
        [absent, present_in_mirror, present_final], Mock()
    ) == [absent]


def stub_source_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Serve the index downloads and GRIB header reads `file_refs` makes from
    `tmp_path/mirror` and `tmp_path/nodd`, chosen by the URL's container prefix."""

    def local_path(url: str) -> Path:
        if url.startswith(MIRROR_LOCATION_PREFIX):
            return tmp_path / "mirror" / url.removeprefix(MIRROR_LOCATION_PREFIX)
        return tmp_path / "nodd" / url.removeprefix(S3_LOCATION_PREFIX)

    def download(url: str, dataset_id: str, **kwargs: object) -> Path:
        copy = tmp_path / f"{next(_copy_counter)}.idx"
        copy.write_bytes(local_path(url).read_bytes())
        return copy

    def read_bytes(url: str, *, start: int, end: int, **kwargs: object) -> bytes:
        return local_path(url).read_bytes()[start:end]

    monkeypatch.setattr(shared_region_job_module, "s3_download_to_disk", download)
    monkeypatch.setattr(shared_region_job_module, "s3_read_bytes", read_bytes)


def test_a_mirror_file_is_ingested_then_repointed_to_nodd_on_the_next_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end against a local icechunk store: fire 1 ingests the file from the
    mirror the tick it appears there; fire 2 rewrites its refs from NODD."""
    mirror_dir, nodd_dir = tmp_path / "mirror", tmp_path / "nodd"
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
    template_ds = dataset.template_config.get_template(
        MIRRORED_INIT + pd.Timedelta("1h")
    )
    template_utils.write_metadata(template_ds, dataset.store_factory)
    (_, repo), *_ = dataset.store_factory.icechunk_repos(sort="primary-first")
    position = int(
        template_ds.to_dataset()
        .get_index("init_time")
        .get_indexer(pd.Index([MIRRORED_INIT]))[0]
    )
    data_vars = [get_var("composite_reflectivity")]
    coord = routed_coord(data_vars)
    mirror_file(tmp_path, coord)
    nodd_dir.mkdir()
    monkeypatch.setattr(
        shared_region_job_module,
        "s3_store",
        lambda bucket_url, region, **kwargs: obstore.store.LocalStore(nodd_dir),
    )
    stub_source_reads(monkeypatch, tmp_path)

    class LocalMirrorJob(NoaaHrrrForecast18HourVirtualRegionJob):
        ticks: ClassVar[int] = 0

        def mirror_store(self) -> obstore.store.ObjectStore:
            return obstore.store.LocalStore(mirror_dir)

    monkeypatch.setattr(LocalMirrorJob, "tick_interval", pd.Timedelta("0s"))

    def fire() -> None:
        job = LocalMirrorJob(
            tmp_store=Path("unused-tmp.zarr"),
            template_ds=template_ds,
            data_vars=data_vars,
            append_dim="init_time",
            region=slice(position, position + 1),
            reformat_job_name="test",
            processing_mode="update",
            repoint_mirrored=True,
            poll_deadline=pd.Timestamp.now() + pd.Timedelta("30s"),
        )
        remaining = job.filter_already_present(
            job.source_file_coords(), repo.readonly_session("main").store
        )
        job.process_virtual(repo, [], "main", remaining)

    mirror_url = MIRROR_LOCATION_PREFIX + mirror_key(coord)
    nodd_url = S3_LOCATION_PREFIX + mirror_key(coord)

    fire()
    locations = repo.readonly_session("main").all_virtual_chunk_locations()
    assert mirror_url in locations
    assert nodd_url not in locations

    target = nodd_dir / mirror_key(coord)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(FIXTURE_GRIB.read_bytes())
    target.with_name(target.name + ".idx").write_bytes(FIXTURE_INDEX.read_bytes())
    fire()
    locations = repo.readonly_session("main").all_virtual_chunk_locations()
    assert nodd_url in locations
    assert mirror_url not in locations
