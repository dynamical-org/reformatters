from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import httpx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from gribberish import (
    parse_grib_array,  # ty: ignore[unresolved-import] - native module member
    parse_grib_message_metadata,  # ty: ignore[unresolved-import] - native module member
)

from reformatters.common import virtual_region_job
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.noaa.gefs.forecast_10_day_spatial import (
    region_job as region_job_module,
)
from reformatters.noaa.gefs.forecast_10_day_spatial.region_job import (
    GefsForecast10DaySpatialRegionJob,
    GefsForecast10DaySpatialSourceFileCoord,
    _vars_in_s_file,
)
from reformatters.noaa.gefs.forecast_10_day_spatial.template_config import (
    GefsForecast10DaySpatialTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

TEMPLATE_CONFIG = GefsForecast10DaySpatialTemplateConfig()


def get_var(name: str) -> GEFSDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.name == name)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    # Spans the 2022-10-18T12 transition when "s+b-b22" vars entered the s files.
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2022-10-19T00:00"))


def make_job(
    template_ds: xr.DataTree,
    data_vars: Sequence[GEFSDataVar] | None = None,
    region: slice = slice(0, 1),
    processing_mode: Literal["backfill", "update"] = "backfill",
) -> GefsForecast10DaySpatialRegionJob:
    return GefsForecast10DaySpatialRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars or TEMPLATE_CONFIG.data_vars,
        append_dim="init_time",
        region=region,
        reformat_job_name="test",
        processing_mode=processing_mode,
    )


def test_source_file_coord_urls() -> None:
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T06:00"),
        ensemble_member=1,
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("temperature_2m")],
    )
    # The single canonical URL is the s3:// source location refs point at,
    # matching the dataset's virtual chunk container prefix.
    assert coord.get_url() == (
        "s3://noaa-gefs-pds/gefs.20240101/06/atmos/pgrb2sp25/gep01.t06z.pgrb2s.0p25.f003"
    )
    assert coord.get_index_url() == (
        "s3://noaa-gefs-pds/gefs.20240101/06/atmos/pgrb2sp25/gep01.t06z.pgrb2s.0p25.f003.idx"
    )


def test_source_file_coord_url_control_member_max_lead() -> None:
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T00:00"),
        ensemble_member=0,
        lead_time=pd.Timedelta("240h"),
        data_vars=[get_var("temperature_2m")],
    )
    assert coord.get_url() == (
        "s3://noaa-gefs-pds/gefs.20240101/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.f240"
    )


def test_source_file_coord_rejects_lead_beyond_s_files() -> None:
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T00:00"),
        ensemble_member=1,
        lead_time=pd.Timedelta("246h"),
        data_vars=[get_var("temperature_2m")],
    )
    with pytest.raises(AssertionError, match="s files end at"):
        _ = coord.gefs_file_type


def test_source_file_coord_rejects_pre_b22_var() -> None:
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2021-01-01T00:00"),  # before the b22 transition
        ensemble_member=1,
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("percent_frozen_precipitation_surface")],  # "s+b-b22"
    )
    with pytest.raises(AssertionError, match="not in the s file"):
        _ = coord.gefs_file_type


def test_generate_source_file_coords_one_file_per_init_member_lead(
    template_ds: xr.DataTree,
) -> None:
    data_vars = [
        get_var("temperature_2m"),  # instant
        get_var("total_precipitation_surface"),  # no hour-0 values
        get_var("percent_frozen_precipitation_surface"),  # "s+b-b22", instant
    ]
    job = make_job(template_ds, data_vars=data_vars)
    # The final init (2022-10-18T18) is after the b22 transition.
    processing_region_ds = (
        template_ds.isel(init_time=slice(-1, None), ensemble_member=slice(0, 2))
        .sel(lead_time=[pd.Timedelta("0h"), pd.Timedelta("3h")])
        .to_dataset()
    )

    coords = job.generate_source_file_coords(processing_region_ds, data_vars)

    # One s file per (init, member, lead): 1 init x 2 members x 2 leads.
    assert len(coords) == 4
    for coord in coords:
        assert coord.gefs_file_type == "s"
        if coord.lead_time == pd.Timedelta("0h"):
            # precipitation has no hour-0 values
            assert [v.name for v in coord.data_vars] == [
                "temperature_2m",
                "percent_frozen_precipitation_surface",
            ]
        else:
            assert [v.name for v in coord.data_vars] == [v.name for v in data_vars]


def test_generate_source_file_coords_excludes_b22_vars_before_transition(
    template_ds: xr.DataTree,
) -> None:
    data_vars = [
        get_var("temperature_2m"),
        get_var("percent_frozen_precipitation_surface"),  # "s+b-b22"
    ]
    job = make_job(template_ds, data_vars=data_vars)
    # The first init (2020-10-01T00) is before the b22 transition.
    processing_region_ds = (
        template_ds.isel(init_time=slice(0, 1), ensemble_member=slice(0, 1))
        .sel(lead_time=[pd.Timedelta("3h")])
        .to_dataset()
    )

    (coord,) = job.generate_source_file_coords(processing_region_ds, data_vars)
    assert [v.name for v in coord.data_vars] == ["temperature_2m"]
    assert coord.gefs_file_type == "s"


def test_representative_var_uses_coords_own_file_vars(template_ds: xr.DataTree) -> None:
    job = make_job(template_ds)
    avg_then_instant = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T00:00"),
        ensemble_member=1,
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("total_precipitation_surface"), get_var("temperature_2m")],
    )
    assert job.representative_var(avg_then_instant).name == "temperature_2m"

    avg_only = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T00:00"),
        ensemble_member=1,
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("total_precipitation_surface")],
    )
    assert job.representative_var(avg_only).name == "total_precipitation_surface"


_INDEX_CONTENT = """1:0:d=2020100100:PRES:surface:3 hour fcst:ENS=+1
2:1000:d=2020100100:TMP:2 m above ground:3 hour fcst:ENS=+1
3:2500:d=2020100100:APCP:surface:0-3 hour acc fcst:ENS=+1
"""

_PREFIX = "gefs.20201001/00/atmos/pgrb2sp25/"


def _coord(
    member: int, data_vars: Sequence[GEFSDataVar]
) -> GefsForecast10DaySpatialSourceFileCoord:
    return GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2020-10-01T00:00"),
        ensemble_member=member,
        lead_time=pd.Timedelta("3h"),
        data_vars=data_vars,
    )


def _fake_index_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        # Unique per url so concurrent file_refs don't clobber each other's file.
        index_path = tmp_path / (url.rsplit("/", 1)[-1] + ".idx")
        index_path.write_text(_INDEX_CONTENT)
        return index_path

    monkeypatch.setattr(region_job_module, "s3_download_to_disk", fake_download)


def _fake_discover(
    monkeypatch: pytest.MonkeyPatch,
    ticks: list[list[tuple[GefsForecast10DaySpatialSourceFileCoord, int]]],
) -> list[int]:
    """Drive the loop: return one canned (coord, size) list per discovery sweep."""
    sweeps: list[int] = []
    it = iter(ticks)

    def fake(
        pending: list[GefsForecast10DaySpatialSourceFileCoord], **kwargs: object
    ) -> list[tuple[GefsForecast10DaySpatialSourceFileCoord, int]]:
        sweeps.append(len(pending))
        return next(it)

    monkeypatch.setattr(
        region_job_module, "discover_available_by_obstore_listing", fake
    )
    return sweeps


def test_process_virtual_refs_backfill_sweeps_once(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_vars = [get_var("temperature_2m"), get_var("total_precipitation_surface")]
    job = make_job(template_ds, data_vars=data_vars)
    _fake_index_download(monkeypatch, tmp_path)
    coord1, coord2 = _coord(1, data_vars), _coord(2, data_vars)
    # Member 1 ready; member 2 not yet.
    sweeps = _fake_discover(monkeypatch, [[(coord1, 9000)]])

    batches = list(job.process_virtual_refs([coord1, coord2]))

    # A backfill sweeps once, yields the ready file, then exits without polling.
    assert len(sweeps) == 1
    (batch,) = batches
    ((coord, refs),) = batch
    assert coord.ensemble_member == 1
    assert [r.data_var.name for r in refs] == [
        "temperature_2m",
        "total_precipitation_surface",
    ]
    tmp_ref, apcp_ref = refs
    assert tmp_ref.offset == 1000
    assert tmp_ref.length == 1500
    # APCP is the last message in the index; its end comes from the listed file size.
    assert apcp_ref.offset == 2500
    assert apcp_ref.length == 9000 - 2500
    member1 = f"{_PREFIX}gep01.t00z.pgrb2s.0p25.f003"
    for ref in refs:
        assert ref.location == f"s3://noaa-gefs-pds/{member1}"
        assert ref.out_loc == {
            "init_time": pd.Timestamp("2020-10-01T00:00"),
            "lead_time": pd.Timedelta("3h"),
            "ensemble_member": 1,
        }


def test_process_virtual_refs_update_polls_until_all_ingested(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars=data_vars, processing_mode="update")
    _fake_index_download(monkeypatch, tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr(virtual_region_job.time, "sleep", sleeps.append)

    coord1, coord2 = _coord(1, data_vars), _coord(2, data_vars)
    # Tick 1: nothing; tick 2: member 1; tick 3: member 2.
    _fake_discover(monkeypatch, [[], [(coord1, 9000)], [(coord2, 9000)]])

    batches = list(job.process_virtual_refs([coord1, coord2]))

    # One yield per tick that found new files; exits once all are ingested.
    assert [
        [
            (coord.ensemble_member, [r.data_var.name for r in refs])
            for coord, refs in batch
        ]
        for batch in batches
    ] == [
        [(1, ["temperature_2m"])],
        [(2, ["temperature_2m"])],
    ]
    # Slept between ticks (after ticks 1 and 2, not after the final tick), never
    # longer than the tick interval and never a negative duration.
    assert len(sleeps) == 2
    assert all(0 <= s <= job.tick_interval.total_seconds() for s in sleeps)


def test_file_refs_skips_file_whose_index_points_past_eof(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars=data_vars)
    _fake_index_download(monkeypatch, tmp_path)  # index's last message starts at 2500
    coord = _coord(1, data_vars)
    # A matching (larger) file resolves refs.
    assert job.file_refs(coord, file_size=9000)
    # A file smaller than the index's max offset is stale/mismatched -> no refs.
    assert job.file_refs(coord, file_size=2000) == []


def test_process_virtual_refs_drops_files_with_stale_index(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars=data_vars)
    _fake_index_download(monkeypatch, tmp_path)
    coord1, coord2 = _coord(1, data_vars), _coord(2, data_vars)
    # member1's size (2000) is below the index's max offset (2500) -> file_refs skips it.
    _fake_discover(monkeypatch, [[(coord1, 2000), (coord2, 9000)]])

    batches = list(job.process_virtual_refs([coord1, coord2]))

    (batch,) = batches
    ((coord, _refs),) = batch
    assert coord.ensemble_member == 2  # member1 dropped, only member2 ingested


def test_process_virtual_refs_update_all_skipped_tick_yields_nothing_and_drops(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # In update mode, a tick where every discovered file is skipped (file_refs
    # returns no refs) yields no batch, and the skipped files are permanently
    # dropped from pending for the rest of the run — later discovery sweeps are
    # not asked about them again. They self-heal on the next cron fire, when
    # filter_already_present re-derives them from the manifest.
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars=data_vars, processing_mode="update")
    _fake_index_download(monkeypatch, tmp_path)
    monkeypatch.setattr(virtual_region_job.time, "sleep", lambda _s: None)

    coord1, coord2 = _coord(1, data_vars), _coord(2, data_vars)
    pending_per_sweep: list[list[GefsForecast10DaySpatialSourceFileCoord]] = []
    # Tick 1: member 1 with a stale index (size 2000 < the index's max offset 2500),
    # so file_refs skips it; tick 2: member 2, ingestible.
    ticks = iter([[(coord1, 2000)], [(coord2, 9000)]])

    def fake(
        pending: list[GefsForecast10DaySpatialSourceFileCoord], **kwargs: object
    ) -> list[tuple[GefsForecast10DaySpatialSourceFileCoord, int]]:
        pending_per_sweep.append(list(pending))
        return next(ticks)

    monkeypatch.setattr(
        region_job_module, "discover_available_by_obstore_listing", fake
    )

    batches = list(job.process_virtual_refs([coord1, coord2]))

    # No batch for the all-skipped tick; only member 2 is ever yielded.
    (batch,) = batches
    ((coord, _refs),) = batch
    assert coord is coord2
    assert [[id(c) for c in pending] for pending in pending_per_sweep] == [
        [id(coord1), id(coord2)],
        [id(coord2)],
    ]


def test_file_refs_or_skip_swallows_unexpected_errors(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(template_ds, data_vars=[get_var("temperature_2m")])
    coord = _coord(1, [get_var("temperature_2m")])

    def boom(self: object, *args: object, **kwargs: object) -> list[VirtualRef]:
        raise RuntimeError("network blip")

    # Patch on the class: file_refs is a public name, which pydantic's __setattr__
    # rejects as a non-field on the model instance.
    monkeypatch.setattr(type(job), "file_refs", boom)
    # An unexpected per-file error skips the file (no refs) rather than raising.
    assert job._file_refs_or_skip(coord, 9000) == []


def test_file_refs_or_skip_propagates_assertion_errors(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(template_ds, data_vars=[get_var("temperature_2m")])
    coord = _coord(1, [get_var("temperature_2m")])

    def bad_invariant(
        self: object, *args: object, **kwargs: object
    ) -> list[VirtualRef]:
        raise AssertionError("our own invariant")

    monkeypatch.setattr(type(job), "file_refs", bad_invariant)
    # Assertion failures are our own bugs and must surface, not be swallowed.
    with pytest.raises(AssertionError, match="our own invariant"):
        job._file_refs_or_skip(coord, 9000)


@pytest.mark.slow
def test_real_source_all_vars_resolve_and_decode(template_ds: xr.DataTree) -> None:
    """Guard against NOAA layout/index drift: list the real bucket, parse a real
    index for every variable, and decode one message to check values + grid."""
    init = pd.Timestamp("2024-01-01T00:00")
    lead = pd.Timedelta("6h")
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=init,
        ensemble_member=1,
        lead_time=lead,
        data_vars=_vars_in_s_file(list(TEMPLATE_CONFIG.data_vars), init, lead),
    )
    job = make_job(template_ds)

    [(_, file_size)] = job.discover_available([coord])
    key = coord.get_url().removeprefix("s3://noaa-gefs-pds/")
    refs = job.file_refs(coord, file_size)

    assert len(refs) == 19
    for ref in refs:
        assert 0 <= ref.offset < file_size
        assert 0 < ref.length
        assert ref.offset + ref.length <= file_size
        assert ref.location == coord.get_url()

    t2m_ref = next(r for r in refs if r.data_var.name == "temperature_2m")
    response = httpx.get(
        f"https://noaa-gefs-pds.s3.amazonaws.com/{key}",
        headers={
            "Range": f"bytes={t2m_ref.offset}-{t2m_ref.offset + t2m_ref.length - 1}"
        },
    )
    response.raise_for_status()
    data = parse_grib_array(response.content, 0)
    assert data.size == 721 * 1440
    assert 180 < np.nanmin(data) < 280 < np.nanmax(data) < 340  # plausible Kelvin

    # The template grid is the codec's decoded grid: latitude north-first (the GEFS
    # message is already north-first) and longitude rewrapped by adjust_longitude_range
    # from the raw 0-360 message grid to a monotonic -180..180.
    lat, lon = parse_grib_message_metadata(response.content, 0).latlng()
    dim_coords = TEMPLATE_CONFIG.dimension_coordinates()
    np.testing.assert_allclose(np.asarray(lat), dim_coords["latitude"])
    rewrapped_lon = np.sort(((np.asarray(lon) + 180) % 360) - 180)
    np.testing.assert_allclose(rewrapped_lon, dim_coords["longitude"])


def test_operational_update_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    now = pd.Timestamp("2020-10-03T01:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = GefsForecast10DaySpatialRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="init_time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )

    (job,) = jobs
    assert isinstance(job, GefsForecast10DaySpatialRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    assert init_times[-1] == pd.Timestamp("2020-10-03T00:00")
    # One job spanning the 24h active window (4 init times at 6h frequency).
    assert job.region == slice(len(init_times) - 4, len(init_times))
    assert [v.name for v in job.data_vars] == [
        v.name for v in TEMPLATE_CONFIG.data_vars
    ]
