from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.hrrr import virtual_region_job as region_job_module
from reformatters.noaa.hrrr.analysis_virtual.region_job import (
    NoaaHrrrAnalysisVirtualRegionJob,
    NoaaHrrrAnalysisVirtualSourceFileCoord,
)
from reformatters.noaa.hrrr.analysis_virtual.template_config import (
    NoaaHrrrAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.virtual_region_job import NoaaHrrrVirtualRegionJob

TEMPLATE_CONFIG = NoaaHrrrAnalysisVirtualTemplateConfig()


def get_var(path: str) -> NoaaHrrrDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.path == path)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2014-10-01T03:00"))


def make_job(
    template_ds: xr.DataTree,
    data_vars: Sequence[NoaaHrrrDataVar],
    region: slice = slice(0, 1),
) -> NoaaHrrrAnalysisVirtualRegionJob:
    return NoaaHrrrAnalysisVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="time",
        region=region,
        reformat_job_name="test",
    )


def fake_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str) -> None:
    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        path = tmp_path / url.rsplit("/", 1)[-1]
        path.write_text(content)
        return path

    monkeypatch.setattr(region_job_module, "s3_download_to_disk", fake_download)


def test_source_file_coord_url_and_out_loc() -> None:
    coord = NoaaHrrrAnalysisVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T06:00"),
        lead_time=pd.Timedelta("1h"),
        domain="conus",
        file_type="sfc",
        data_vars=[get_var("total_precipitation_surface")],
    )
    assert coord.get_url() == (
        "s3://noaa-hrrr-bdp-pds/hrrr.20240601/conus/hrrr.t06z.wrfsfcf01.grib2"
    )
    assert coord.get_index_url() == coord.get_url() + ".idx"
    assert dict(coord.out_loc()) == {"time": pd.Timestamp("2024-06-01T07:00")}


def test_group_file_probe_loc_carries_first_level(template_ds: xr.DataTree) -> None:
    # A prs/nat file holds only group vars, so the per-file manifest probe supplements
    # a concrete level to resolve to a single chunk; out_loc itself stays the file's slab.
    prs = NoaaHrrrAnalysisVirtualSourceFileCoord(
        init_time=pd.Timestamp("2014-10-01T01:00"),
        lead_time=pd.Timedelta("0h"),
        domain="conus",
        file_type="prs",
        data_vars=[get_var("pressure_level/temperature")],
    )
    assert dict(prs.out_loc()) == {"time": pd.Timestamp("2014-10-01T01:00")}
    job = make_job(template_ds, data_vars=list(prs.data_vars))
    probe_loc = job.representative_probe_loc(prs, job.representative_var(prs))
    assert dict(probe_loc) == {
        "time": pd.Timestamp("2014-10-01T01:00"),
        "pressure_level": 1000,
    }


def test_truncated_source_files_are_never_available(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2016-08-05 native uploads that end mid-file never reach the write loop."""
    data_vars = [get_var("model_level/temperature")]
    coords = [
        NoaaHrrrAnalysisVirtualSourceFileCoord(
            init_time=pd.Timestamp(f"2016-08-05T{hour:02d}:00"),
            lead_time=pd.Timedelta(0),
            domain="conus",
            file_type="nat",
            data_vars=data_vars,
        )
        for hour in (9, 10, 12)
    ]
    monkeypatch.setattr(
        region_job_module,
        "discover_available_by_obstore_listing",
        lambda pending, **kwargs: [(coord, 100) for coord in pending],
    )
    job = make_job(template_ds, data_vars=data_vars)

    assert [coord.get_url() for coord, _ in job.discover_available(coords)] == [
        "s3://noaa-hrrr-bdp-pds/hrrr.20160805/conus/hrrr.t09z.wrfnatf00.grib2"
    ]


@pytest.mark.parametrize(
    ("path", "element", "level"),
    [
        ("pressure_level/cloud_ice_mixing_ratio", "CICE", "1000 mb"),
        ("pressure_level/cloud_ice_mixing_ratio", "CIMIXR", "1000 mb"),
        ("model_level/cloud_ice_mixing_ratio", "CICE", "1 hybrid level"),
        ("model_level/cloud_ice_mixing_ratio", "CIMIXR", "1 hybrid level"),
    ],
)
def test_cloud_ice_index_spellings_emit_refs(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
    element: str,
    level: str,
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        f"1:0:d=2024060100:{element}:{level}:anl:\n",
    )
    var = get_var(path)
    file_type = "prs" if var.group == "pressure_level" else "nat"
    coord = NoaaHrrrAnalysisVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T00:00"),
        lead_time=pd.Timedelta(0),
        domain="conus",
        file_type=file_type,
        data_vars=[var],
    )
    job = make_job(template_ds, data_vars=[var])

    refs = job.file_refs(coord, file_size=1000)

    assert [(ref.data_var.path, ref.offset, ref.length) for ref in refs] == [
        (path, 0, 1000)
    ]


def test_generate_source_file_coords_shortest_available_lead(
    template_ds: xr.DataTree,
) -> None:
    data_vars = [
        get_var("temperature_2m"),  # sfc, has hour-0 values
        get_var("total_precipitation_surface"),  # sfc, accum (no hour 0)
        get_var("pressure_level/temperature"),  # prs, has hour-0 values
    ]
    job = make_job(template_ds, data_vars=data_vars, region=slice(0, 2))
    region_ds = template_ds.to_dataset().isel(time=slice(0, 2))

    coords = job.generate_source_file_coords(region_ds, data_vars)

    by_key = {(c.file_type, c.lead_time, c.out_loc()["time"]): c for c in coords}
    assert len(coords) == 6  # (sfc f00, sfc f01, prs f00) x 2 times
    time = pd.Timestamp("2014-10-01T00:00")

    hour_0_coord = by_key[("sfc", pd.Timedelta("0h"), time)]
    assert hour_0_coord.init_time == time
    assert {v.name for v in hour_0_coord.data_vars} == {"temperature_2m"}

    hour_1_coord = by_key[("sfc", pd.Timedelta("1h"), time)]
    assert hour_1_coord.init_time == time - pd.Timedelta("1h")
    assert {v.name for v in hour_1_coord.data_vars} == {"total_precipitation_surface"}

    prs_coord = by_key[("prs", pd.Timedelta("0h"), time)]
    assert prs_coord.init_time == time
    assert [v.path for v in prs_coord.data_vars] == ["pressure_level/temperature"]


def test_full_catalog_sources_four_files_per_time(template_ds: xr.DataTree) -> None:
    """Every variable is sourced once, from the shortest lead that carries it."""
    data_vars = TEMPLATE_CONFIG.data_vars
    job = make_job(template_ds, data_vars=data_vars)
    region_ds = template_ds.to_dataset().isel(time=slice(0, 1))

    coords = job.generate_source_file_coords(region_ds, data_vars)

    assert {(c.file_type, c.lead_time) for c in coords} == {
        ("sfc", pd.Timedelta("0h")),
        ("sfc", pd.Timedelta("1h")),
        ("prs", pd.Timedelta("0h")),
        ("nat", pd.Timedelta("0h")),
    }
    sourced = [var for coord in coords for var in coord.data_vars]
    # The region starts before every analysis_usable_from boundary, so those vars are not sourced.
    gated = {
        v.path for v in data_vars if v.internal_attrs.analysis_usable_from is not None
    }
    assert gated
    assert sorted(v.path for v in sourced) == sorted(
        v.path for v in data_vars if v.path not in gated
    )
    for coord in coords:
        from_f00 = coord.lead_time == pd.Timedelta("0h")
        assert all(v.has_hour_0_values() == from_f00 for v in coord.data_vars)


def test_representative_var_is_available_in_every_era(
    template_ds: xr.DataTree,
) -> None:
    """The probe variable is never one the source began publishing partway through."""
    data_vars = TEMPLATE_CONFIG.data_vars
    job = make_job(template_ds, data_vars=data_vars)
    region_ds = template_ds.to_dataset().isel(time=slice(0, 1))

    coords = job.generate_source_file_coords(region_ds, data_vars)

    assert {
        (c.file_type, c.lead_time): job.representative_var(c).path for c in coords
    } == {
        ("sfc", pd.Timedelta("0h")): "composite_reflectivity",
        ("sfc", pd.Timedelta("1h")): "categorical_rain_surface",
        ("prs", pd.Timedelta("0h")): "pressure_level/temperature",
        ("nat", pd.Timedelta("0h")): "model_level/temperature",
    }
    for coord in coords:
        assert job.representative_var(coord).internal_attrs.analysis_usable_from is None


def test_a_variable_is_not_sourced_before_its_analysis_usable_from(
    template_ds: xr.DataTree,
) -> None:
    """The file is still read for its other variables; only the gated one drops out."""
    tke = get_var("model_level/turbulent_kinetic_energy")
    mate = get_var("model_level/temperature")
    analysis_usable_from = tke.internal_attrs.analysis_usable_from
    assert analysis_usable_from is not None
    data_vars = [tke, mate]
    job = make_job(template_ds, data_vars=data_vars)

    for time, expected in (
        (analysis_usable_from - pd.Timedelta("1h"), {mate.path}),
        (analysis_usable_from, {mate.path, tke.path}),
    ):
        region_ds = xr.Dataset(coords={"time": pd.to_datetime([time])})
        coords = job.generate_source_file_coords(region_ds, data_vars)
        assert len(coords) == 1, time
        assert {v.path for v in coords[0].data_vars} == expected, time


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2024-06-02T01:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = NoaaHrrrAnalysisVirtualRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )

    (job,) = jobs
    assert isinstance(job, NoaaHrrrAnalysisVirtualRegionJob)
    assert job.processing_mode == "update"
    times = template_ds.to_dataset().get_index("time")
    assert job.region == slice(len(times) - 12, len(times))


def hour_0_and_hour_1_coords(
    template_ds: xr.DataTree, times: Sequence[pd.Timestamp]
) -> list[NoaaHrrrAnalysisVirtualSourceFileCoord]:
    data_vars = [
        get_var("temperature_2m"),  # sfc, has hour-0 values
        get_var("total_precipitation_surface"),  # sfc, accum (no hour 0)
    ]
    job = make_job(template_ds, data_vars=data_vars)
    region_ds = template_ds.to_dataset().sel(time=list(times))
    return list(job.generate_source_file_coords(region_ds, data_vars))


def discover(
    job: NoaaHrrrAnalysisVirtualRegionJob,
    pending: Sequence[NoaaHrrrAnalysisVirtualSourceFileCoord],
    published: Sequence[NoaaHrrrAnalysisVirtualSourceFileCoord],
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[pd.Timedelta, pd.Timestamp]]:
    """Run the gate over `pending` with `published` listed by the source."""
    monkeypatch.setattr(
        NoaaHrrrVirtualRegionJob,
        "discover_available",
        lambda self, pending: [(c, 100) for c in pending if c in published],
    )
    return [
        (coord.lead_time, coord.valid_time())
        for coord, _ in job.discover_available(list(pending))
    ]


def test_hour_1_file_withheld_until_the_hour_0_file_publishes(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    time = pd.Timestamp("2014-10-01T02:00")
    coords = hour_0_and_hour_1_coords(template_ds, [time])
    (hour_0,) = [c for c in coords if c.lead_time == pd.Timedelta("0h")]
    (hour_1,) = [c for c in coords if c.lead_time == pd.Timedelta("1h")]
    job = make_job(
        template_ds, data_vars=list(hour_0.data_vars) + list(hour_1.data_vars)
    ).model_copy(update={"ingested_through": time - pd.Timedelta("1h")})

    # f01 publishes an hour before the f00 that authorizes the same valid time.
    assert discover(job, coords, [hour_1], monkeypatch) == []
    assert sorted(discover(job, coords, [hour_1, hour_0], monkeypatch)) == [
        (pd.Timedelta("0h"), time),
        (pd.Timedelta("1h"), time),
    ]


def test_only_the_hour_with_its_hour_0_file_is_released(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tick carrying two hours' files extends to the newer hour only if it is whole."""
    times = [pd.Timestamp("2014-10-01T01:00"), pd.Timestamp("2014-10-01T02:00")]
    coords = hour_0_and_hour_1_coords(template_ds, times)
    published = [
        c
        for c in coords
        if c.valid_time() == times[0] or c.lead_time == pd.Timedelta("1h")
    ]
    job = make_job(template_ds, data_vars=list(coords[0].data_vars)).model_copy(
        update={"ingested_through": times[0] - pd.Timedelta("1h")}
    )

    assert sorted(discover(job, coords, published, monkeypatch)) == [
        (pd.Timedelta("0h"), times[0]),
        (pd.Timedelta("1h"), times[0]),
    ]


def test_a_time_the_store_already_covers_is_never_withheld(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An f00 the archive never published must not block the f01 beside it forever."""
    time = pd.Timestamp("2014-10-01T02:00")
    coords = hour_0_and_hour_1_coords(template_ds, [time])
    (hour_1,) = [c for c in coords if c.lead_time == pd.Timedelta("1h")]
    job = make_job(template_ds, data_vars=list(hour_1.data_vars)).model_copy(
        update={"ingested_through": time}
    )

    assert discover(job, coords, [hour_1], monkeypatch) == [(pd.Timedelta("1h"), time)]


def test_an_empty_store_waits_for_an_hour_0_file(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    time = pd.Timestamp("2014-10-01T02:00")
    coords = hour_0_and_hour_1_coords(template_ds, [time])
    (hour_1,) = [c for c in coords if c.lead_time == pd.Timedelta("1h")]
    job = make_job(template_ds, data_vars=list(hour_1.data_vars))
    assert job.ingested_through is None

    assert discover(job, coords, [hour_1], monkeypatch) == []
