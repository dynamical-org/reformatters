from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.hrrr.analysis_virtual.region_job import (
    NoaaHrrrAnalysisVirtualRegionJob,
    NoaaHrrrAnalysisVirtualSourceFileCoord,
)
from reformatters.noaa.hrrr.analysis_virtual.template_config import (
    NoaaHrrrAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar

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
    assert sorted(v.path for v in sourced) == sorted(v.path for v in data_vars)
    for coord in coords:
        from_f00 = coord.lead_time == pd.Timedelta("0h")
        assert all(v.has_hour_0_values() == from_f00 for v in coord.data_vars)


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
