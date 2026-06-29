from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.hrrr.forecast_48_hour_spatial import (
    region_job as region_job_module,
)
from reformatters.noaa.hrrr.forecast_48_hour_spatial.region_job import (
    NoaaHrrrForecast48HourSpatialRegionJob,
    NoaaHrrrForecast48HourSpatialSourceFileCoord,
)
from reformatters.noaa.hrrr.forecast_48_hour_spatial.template_config import (
    NoaaHrrrForecast48HourSpatialTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar

TEMPLATE_CONFIG = NoaaHrrrForecast48HourSpatialTemplateConfig()
_LEAD_6H = pd.Timedelta("6h")


def get_var(path: str) -> NoaaHrrrDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.path == path)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2018-07-14T00:00"))


def make_job(
    template_ds: xr.DataTree,
    data_vars: Sequence[NoaaHrrrDataVar] | None = None,
    region: slice = slice(0, 1),
    processing_mode: Literal["backfill", "update"] = "backfill",
) -> NoaaHrrrForecast48HourSpatialRegionJob:
    return NoaaHrrrForecast48HourSpatialRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars or TEMPLATE_CONFIG.data_vars,
        append_dim="init_time",
        region=region,
        reformat_job_name="test",
        processing_mode=processing_mode,
    )


def _coord(
    file_type: Literal["sfc", "prs", "nat"],
    data_vars: Sequence[NoaaHrrrDataVar],
    lead_time: pd.Timedelta = _LEAD_6H,
) -> NoaaHrrrForecast48HourSpatialSourceFileCoord:
    return NoaaHrrrForecast48HourSpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T00:00"),
        lead_time=lead_time,
        domain="conus",
        file_type=file_type,
        data_vars=data_vars,
    )


def _fake_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str) -> None:
    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        path = tmp_path / (url.rsplit("/", 1)[-1])
        path.write_text(content)
        return path

    monkeypatch.setattr(region_job_module, "s3_download_to_disk", fake_download)


# --- URLs and out_loc ---


def test_source_file_coord_url_and_index() -> None:
    coord = _coord("sfc", [get_var("temperature_2m")])
    assert coord.get_url() == (
        "s3://noaa-hrrr-bdp-pds/hrrr.20240601/conus/hrrr.t00z.wrfsfcf06.grib2"
    )
    assert coord.get_index_url() == coord.get_url() + ".idx"


def test_out_loc_root_file_excludes_level() -> None:
    coord = _coord("sfc", [get_var("temperature_2m")])
    assert dict(coord.out_loc()) == {
        "init_time": pd.Timestamp("2024-06-01T00:00"),
        "lead_time": pd.Timedelta("6h"),
    }


def test_out_loc_group_file_carries_representative_level() -> None:
    # A prs/nat file holds only group vars, so the per-file probe needs a concrete level.
    prs = _coord("prs", [get_var("pressure_level/temperature")])
    assert dict(prs.out_loc())["pressure_level"] == 1000
    nat = _coord("nat", [get_var("model_level/temperature")])
    assert dict(nat.out_loc())["model_level"] == 1


# --- message-driven file_refs ---

_SFC_INDEX = (
    "1:0:d=2024060100:REFC:entire atmosphere:6 hour fcst:\n"
    "2:500:d=2024060100:TMP:2 m above ground:6 hour fcst:\n"
    "3:1500:d=2024060100:var discipline=0 center=7 local_table=1 parmcat=16 parm=201:entire atmosphere:6 hour fcst:\n"
    "4:2000:d=2024060100:APCP:surface:0-6 hour acc fcst:\n"
    "5:3000:d=2024060100:APCP:surface:5-6 hour acc fcst:\n"
)


def test_file_refs_root_window_disambiguation_and_skips_unmatched(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_index(monkeypatch, tmp_path, _SFC_INDEX)
    data_vars = [
        get_var("temperature_2m"),
        get_var("total_precipitation_run_total_surface"),  # 0-6 hour acc window
        get_var("total_precipitation_surface"),  # 5-6 hour acc window
    ]
    job = make_job(template_ds, data_vars=data_vars)
    refs = job.file_refs(_coord("sfc", data_vars), file_size=9000)

    by_name = {r.data_var.name: r for r in refs}
    # REFC and the unnamed experimental message are not in data_vars -> not emitted.
    assert set(by_name) == {
        "temperature_2m",
        "total_precipitation_run_total_surface",
        "total_precipitation_surface",
    }
    assert (by_name["temperature_2m"].offset, by_name["temperature_2m"].length) == (
        500,
        1000,
    )
    # The two APCP windows route to two distinct variables by window string.
    run_total = by_name["total_precipitation_run_total_surface"]
    one_hour = by_name["total_precipitation_surface"]
    assert (run_total.offset, run_total.length) == (2000, 1000)
    assert (one_hour.offset, one_hour.length) == (3000, 9000 - 3000)
    for ref in refs:
        assert ref.out_loc == {
            "init_time": pd.Timestamp("2024-06-01T00:00"),
            "lead_time": pd.Timedelta("6h"),
        }
        assert ref.location == (
            "s3://noaa-hrrr-bdp-pds/hrrr.20240601/conus/hrrr.t00z.wrfsfcf06.grib2"
        )


def test_file_refs_f01_run_total_and_hourly_share_window(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # At lead 1h the run-total accumulation window (0->1) and the per-hour bucket
    # (0->1) render the identical idx window string, so a single APCP message must
    # populate BOTH the run-total and per-hour variables (one-to-many lookup). Before
    # the fix the per-hour variant overwrote the run-total in the lookup dict and the
    # run-total var got no ref, reading back NaN at f01 for every init.
    _fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2024060100:APCP:surface:0-1 hour acc fcst:\n",
    )
    data_vars = [
        get_var("total_precipitation_run_total_surface"),
        get_var("total_precipitation_surface"),
    ]
    job = make_job(template_ds, data_vars=data_vars)
    refs = job.file_refs(
        _coord("sfc", data_vars, lead_time=pd.Timedelta("1h")), file_size=1000
    )

    assert {r.data_var.name for r in refs} == {
        "total_precipitation_run_total_surface",
        "total_precipitation_surface",
    }
    # Both refs point at the same single message's byte range.
    for ref in refs:
        assert (ref.offset, ref.length) == (0, 1000)


def test_file_refs_pressure_group_routes_each_level(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = (
        "1:0:d=2024060100:TMP:500 mb:6 hour fcst:\n"
        "2:1200:d=2024060100:TMP:50 mb:6 hour fcst:\n"
    )
    _fake_index(monkeypatch, tmp_path, index)
    var = get_var("pressure_level/temperature")
    job = make_job(template_ds, data_vars=[var])
    refs = job.file_refs(_coord("prs", [var]), file_size=2500)

    assert len(refs) == 2
    by_level = {ref.out_loc["pressure_level"]: ref for ref in refs}
    assert (by_level[500].offset, by_level[500].length) == (0, 1200)
    assert (by_level[50].offset, by_level[50].length) == (1200, 2500 - 1200)
    for ref in refs:
        assert ref.data_var.path == "pressure_level/temperature"
        assert ref.out_loc["init_time"] == pd.Timestamp("2024-06-01T00:00")


def test_file_refs_model_group_routes_hybrid_levels(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = (
        "1:0:d=2024060100:TMP:1 hybrid level:6 hour fcst:\n"
        "2:1200:d=2024060100:TMP:50 hybrid level:6 hour fcst:\n"
    )
    _fake_index(monkeypatch, tmp_path, index)
    var = get_var("model_level/temperature")
    job = make_job(template_ds, data_vars=[var])
    refs = job.file_refs(_coord("nat", [var]), file_size=2500)

    assert {ref.out_loc["model_level"] for ref in refs} == {1, 50}


def test_file_refs_skips_stale_index_past_eof(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_index(monkeypatch, tmp_path, _SFC_INDEX)
    data_vars = [get_var("temperature_2m")]  # index says bytes 500..1500
    job = make_job(template_ds, data_vars=data_vars)
    # file truncated below the matched message's end byte -> stale/mismatched -> skip.
    assert job.file_refs(_coord("sfc", data_vars), file_size=1200) == []


def test_file_refs_lead_0_instant_uses_anl_window(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # At lead 0 instant fields use the "anl" window string in the idx.
    _fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2024060100:TMP:2 m above ground:anl:\n",
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars=data_vars)
    refs = job.file_refs(
        _coord("sfc", data_vars, lead_time=pd.Timedelta("0h")), file_size=1000
    )
    assert [r.data_var.name for r in refs] == ["temperature_2m"]


# --- generate_source_file_coords ---


def test_generate_source_file_coords_splits_by_product_and_drops_hour0_accum(
    template_ds: xr.DataTree,
) -> None:
    data_vars = [
        get_var("temperature_2m"),  # sfc, instant
        get_var("total_precipitation_surface"),  # sfc, accum (no hour 0)
        get_var("pressure_level/temperature"),  # prs
        get_var("model_level/temperature"),  # nat
    ]
    job = make_job(template_ds, data_vars=data_vars)
    region_ds = (
        template_ds.to_dataset()
        .isel(init_time=slice(0, 1))
        .sel(lead_time=[pd.Timedelta("0h"), pd.Timedelta("6h")])
    )

    coords = job.generate_source_file_coords(region_ds, data_vars)
    by_key = {(c.file_type, c.lead_time): c for c in coords}

    # 3 products x 2 leads, except sfc at lead 0 keeps only the instant var.
    assert len(coords) == 6
    assert {v.name for v in by_key[("sfc", pd.Timedelta("0h"))].data_vars} == {
        "temperature_2m"
    }
    assert {v.name for v in by_key[("sfc", pd.Timedelta("6h"))].data_vars} == {
        "temperature_2m",
        "total_precipitation_surface",
    }
    # Group vars are instant, so present at lead 0.
    assert by_key[("prs", pd.Timedelta("0h"))].data_vars[0].path == (
        "pressure_level/temperature"
    )


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2024-06-02T01:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = NoaaHrrrForecast48HourSpatialRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="init_time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )
    (job,) = jobs
    assert isinstance(job, NoaaHrrrForecast48HourSpatialRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    # 14h window at the 6h cadence = the current + 2 prior cycles.
    assert job.region == slice(len(init_times) - 3, len(init_times))
