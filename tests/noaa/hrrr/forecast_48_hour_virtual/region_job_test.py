from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa import (
    noaa_virtual_region_job as shared_region_job_module,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.region_job import (
    NoaaHrrrForecast48HourVirtualRegionJob,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.template_config import (
    NoaaHrrrForecast48HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
)

TEMPLATE_CONFIG = NoaaHrrrForecast48HourVirtualTemplateConfig()
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
) -> NoaaHrrrForecast48HourVirtualRegionJob:
    return NoaaHrrrForecast48HourVirtualRegionJob(
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
) -> NoaaHrrrForecastVirtualSourceFileCoord:
    return NoaaHrrrForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T00:00"),
        lead_time=lead_time,
        domain="conus",
        file_type=file_type,
        data_vars=data_vars,
    )


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


def test_group_file_probe_loc_carries_first_level(template_ds: xr.DataTree) -> None:
    # out_loc stays the file's slab; the manifest probe supplements a concrete level.
    prs = _coord("prs", [get_var("pressure_level/temperature")])
    assert "pressure_level" not in prs.out_loc()
    job = make_job(template_ds, data_vars=prs.data_vars)
    prs_probe = job.representative_probe_loc(prs, job.representative_var(prs))
    assert prs_probe["pressure_level"] == 1000

    nat = _coord("nat", [get_var("model_level/temperature")])
    nat_probe = job.representative_probe_loc(nat, job.representative_var(nat))
    assert nat_probe["model_level"] == 1


# --- discover_available ---


def test_discover_available_lists_source_bucket_requiring_index(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_vars = [get_var("temperature_2m")]
    coord = _coord("sfc", data_vars)
    captured: dict[str, object] = {}

    def fake(
        pending: list[NoaaHrrrForecastVirtualSourceFileCoord], **kwargs: object
    ) -> list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]:
        captured.update(kwargs)
        return [(pending[0], 9000)]

    monkeypatch.setattr(
        shared_region_job_module, "discover_available_by_obstore_listing", fake
    )
    job = make_job(template_ds, data_vars=data_vars)

    result = job.discover_available([coord])

    assert len(result) == 1
    assert result[0][0] is coord
    # HRRR data files always land with a .idx sidecar; a file isn't ready until both exist.
    assert captured["require_index"] is True
    assert captured["location_prefix"] == "s3://noaa-hrrr-bdp-pds/"


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

    jobs, template_ds = NoaaHrrrForecast48HourVirtualRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="init_time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )
    (job,) = jobs
    assert isinstance(job, NoaaHrrrForecast48HourVirtualRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    # The template extends to "now": the last init is within one cycle of it.
    assert init_times[-1] <= now
    assert now - init_times[-1] < TEMPLATE_CONFIG.append_dim_frequency
    # 14h window at the 6h cadence = the current + 2 prior cycles.
    assert job.region == slice(len(init_times) - 3, len(init_times))
