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

from reformatters.common.virtual_region_job import VirtualRef
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ifs_ens.forecast_15_day_spatial import (
    region_job as region_job_module,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_spatial.region_job import (
    _S3_LOCATION_PREFIX,
    EcmwfIfsEnsForecast15DaySpatialRegionJob,
    IfsEnsForecast15DaySpatialSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_spatial.template_config import (
    EcmwfIfsEnsForecast15DaySpatialTemplateConfig,
)

TEMPLATE_CONFIG = EcmwfIfsEnsForecast15DaySpatialTemplateConfig()

_CONTROL = (0,)
_PERTURBED = tuple(range(1, 51))
_LEAD_3H = pd.Timedelta("3h")


def get_var(name: str) -> EcmwfDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.name == name)


@pytest.fixture(scope="module")
def template_ds() -> xr.Dataset:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2026-05-15T00:00"))


def make_job(
    template_ds: xr.Dataset,
    data_vars: Sequence[EcmwfDataVar] | None = None,
    region: slice = slice(0, 1),
    processing_mode: Literal["backfill", "update"] = "backfill",
) -> EcmwfIfsEnsForecast15DaySpatialRegionJob:
    return EcmwfIfsEnsForecast15DaySpatialRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars or TEMPLATE_CONFIG.data_vars,
        append_dim="init_time",
        region=region,
        reformat_job_name="test",
        processing_mode=processing_mode,
    )


def _coord(
    members: tuple[int, ...],
    data_vars: Sequence[EcmwfDataVar],
    lead_time: pd.Timedelta = _LEAD_3H,
) -> IfsEnsForecast15DaySpatialSourceFileCoord:
    return IfsEnsForecast15DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2026-05-13T00:00"),
        lead_time=lead_time,
        ensemble_members=members,
        data_vars=data_vars,
    )


def test_source_file_coord_perturbed_url() -> None:
    coord = _coord(_PERTURBED, [get_var("temperature_2m")])
    assert coord.get_url() == (
        "s3://ecmwf-forecasts/20260513/00z/ifs/0p25/enfo/20260513000000-3h-enfo-ef.grib2"
    )
    assert coord.get_index_url() == (
        "s3://ecmwf-forecasts/20260513/00z/ifs/0p25/enfo/20260513000000-3h-enfo-ef.index"
    )
    assert coord.out_loc()["ensemble_member"] == 1


def test_source_file_coord_control_url() -> None:
    coord = _coord(_CONTROL, [get_var("temperature_2m")])
    assert coord.is_control
    assert coord.get_url() == (
        "s3://ecmwf-forecasts/20260513/00z/ifs/0p25/oper/20260513000000-3h-oper-fc.grib2"
    )
    assert coord.out_loc()["ensemble_member"] == 0


def test_generate_source_file_coords_control_and_perturbed(
    template_ds: xr.Dataset,
) -> None:
    data_vars = [get_var("temperature_2m"), get_var("wind_gust_10m")]
    job = make_job(template_ds, data_vars=data_vars)
    processing_region_ds = template_ds.isel(init_time=slice(0, 1)).sel(
        lead_time=[pd.Timedelta("0h"), pd.Timedelta("3h")]
    )

    coords = job.generate_source_file_coords(processing_region_ds, data_vars)

    # Two files (control oper-fc + perturbed enfo-ef) per (init, lead): 1 init x 2 leads.
    assert len(coords) == 4
    members = {c.ensemble_members for c in coords}
    assert members == {_CONTROL, _PERTURBED}
    for coord in coords:
        if coord.lead_time == pd.Timedelta("0h"):
            # wind_gust is step_type max -> no value at lead 0.
            assert [v.name for v in coord.data_vars] == ["temperature_2m"]
        else:
            assert [v.name for v in coord.data_vars] == [
                "temperature_2m",
                "wind_gust_10m",
            ]


def test_representative_var_prefers_instant(template_ds: xr.Dataset) -> None:
    job = make_job(template_ds)
    accum_then_instant = _coord(
        _PERTURBED, [get_var("total_precipitation_surface"), get_var("temperature_2m")]
    )
    assert job.representative_var(accum_then_instant).name == "temperature_2m"

    accum_only = _coord(_PERTURBED, [get_var("total_precipitation_surface")])
    assert job.representative_var(accum_only).name == "total_precipitation_surface"


# One index file packs every member's messages. 2t and sp for members 1 and 2.
_PERTURBED_INDEX = """\
{"type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "1", "param": "2t", "_offset": 0, "_length": 100}
{"type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "1", "param": "sp", "_offset": 100, "_length": 200}
{"type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "2", "param": "2t", "_offset": 300, "_length": 100}
{"type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "2", "param": "sp", "_offset": 400, "_length": 150}
"""

_ENFO_PREFIX = "20260513/00z/ifs/0p25/enfo/"
_ENFO_FILE = f"{_ENFO_PREFIX}20260513000000-3h-enfo-ef.grib2"


def _fake_index_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_download(url: str, dataset_id: str, *, region: str) -> Path:
        index_path = tmp_path / (url.rsplit("/", 1)[-1])
        index_path.write_text(_PERTURBED_INDEX)
        return index_path

    monkeypatch.setattr(region_job_module, "s3_download_to_disk", fake_download)


def test_file_refs_resolves_all_members(
    template_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_vars = [get_var("temperature_2m"), get_var("pressure_surface")]
    job = make_job(template_ds, data_vars=data_vars)
    _fake_index_download(monkeypatch, tmp_path)
    coord = _coord((1, 2), data_vars)

    refs = job._file_refs(coord, file_size=9000)

    # Two members x two vars, each pointing at the one shared enfo-ef file.
    assert len(refs) == 4
    by_member: dict[int, list[VirtualRef]] = {1: [], 2: []}
    for ref in refs:
        assert ref.location == f"{_S3_LOCATION_PREFIX}{_ENFO_FILE}"
        member = ref.out_loc["ensemble_member"]
        assert isinstance(member, int)
        by_member[member].append(ref)
    m1_t2m, m1_sp = by_member[1]
    assert (m1_t2m.data_var.name, m1_t2m.offset, m1_t2m.length) == (
        "temperature_2m",
        0,
        100,
    )
    assert (m1_sp.data_var.name, m1_sp.offset, m1_sp.length) == (
        "pressure_surface",
        100,
        200,
    )
    m2_t2m, _m2_sp = by_member[2]
    assert (m2_t2m.offset, m2_t2m.length) == (300, 100)


def test_file_refs_skips_file_whose_index_points_past_eof(
    template_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_vars = [get_var("temperature_2m"), get_var("pressure_surface")]
    job = make_job(template_ds, data_vars=data_vars)
    _fake_index_download(monkeypatch, tmp_path)
    coord = _coord((1, 2), data_vars)
    # A matching (larger) file resolves refs.
    assert job._file_refs(coord, file_size=9000)
    # A file smaller than the index's max offset is stale/mismatched -> no refs.
    assert job._file_refs(coord, file_size=400) == []


def _fake_list(monkeypatch: pytest.MonkeyPatch, listing: dict[str, int]) -> list[str]:
    listings: list[str] = []

    def fake_list_objects(prefix: str) -> dict[str, int]:
        listings.append(prefix)
        return listing

    monkeypatch.setattr(region_job_module, "_list_objects", fake_list_objects)
    return listings


def test_discover_available_requires_data_and_index(
    template_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars=data_vars)
    index_key = _ENFO_FILE.removesuffix(".grib2") + ".index"
    enfo2 = f"{_ENFO_PREFIX}20260513000000-6h-enfo-ef.grib2"
    _fake_list(
        monkeypatch,
        {_ENFO_FILE: 9000, index_key: 200, enfo2: 9000},  # enfo2 has no .index yet
    )
    pending = {
        _ENFO_FILE: _coord(_PERTURBED, data_vars),
        enfo2: _coord(_PERTURBED, data_vars, lead_time=pd.Timedelta("6h")),
    }
    assert job._discover_available(pending) == {_ENFO_FILE: 9000}


def test_process_virtual_refs_backfill_sweeps_once(
    template_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_vars = [get_var("temperature_2m"), get_var("pressure_surface")]
    job = make_job(template_ds, data_vars=data_vars)
    _fake_index_download(monkeypatch, tmp_path)
    index_key = _ENFO_FILE.removesuffix(".grib2") + ".index"
    enfo2 = f"{_ENFO_PREFIX}20260513000000-6h-enfo-ef.grib2"
    # File 1 fully published; file 2's .index has not landed yet.
    listings = _fake_list(monkeypatch, {_ENFO_FILE: 9000, index_key: 200, enfo2: 9000})

    coord1 = _coord((1, 2), data_vars)
    coord2 = _coord((1, 2), data_vars, lead_time=pd.Timedelta("6h"))
    batches = list(job.process_virtual_refs([coord1, coord2]))

    # A backfill sweeps once: one listing, one yield with only the ready file, exit.
    assert listings == [_ENFO_PREFIX]
    (batch,) = batches
    ((coord, refs),) = batch
    assert coord.lead_time == pd.Timedelta("3h")
    assert len(refs) == 2 * len(data_vars)  # two members


def test_process_virtual_refs_update_polls_until_all_ingested(
    template_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_vars = [get_var("temperature_2m"), get_var("pressure_surface")]
    job = make_job(template_ds, data_vars=data_vars, processing_mode="update")
    _fake_index_download(monkeypatch, tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr(region_job_module.time, "sleep", sleeps.append)

    index_key = _ENFO_FILE.removesuffix(".grib2") + ".index"
    enfo2 = f"{_ENFO_PREFIX}20260513000000-6h-enfo-ef.grib2"
    enfo2_index = enfo2.removesuffix(".grib2") + ".index"
    # Tick 1: nothing; tick 2: file 1; tick 3: file 2 as well.
    listings = iter(
        [
            {},
            {_ENFO_FILE: 9000, index_key: 200},
            {enfo2: 9000, enfo2_index: 200},
        ]
    )
    monkeypatch.setattr(
        region_job_module, "_list_objects", lambda prefix: next(listings)
    )

    coord1 = _coord((1, 2), data_vars)
    coord2 = _coord((1, 2), data_vars, lead_time=pd.Timedelta("6h"))
    batches = list(job.process_virtual_refs([coord1, coord2]))

    assert [[coord.lead_time for coord, _refs in batch] for batch in batches] == [
        [pd.Timedelta("3h")],
        [pd.Timedelta("6h")],
    ]
    # Slept between ticks (after ticks 1 and 2, not after the final tick).
    assert len(sleeps) == 2


def test_operational_update_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    now = pd.Timestamp("2026-05-16T08:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = (
        EcmwfIfsEnsForecast15DaySpatialRegionJob.operational_update_jobs(
            primary_store=Mock(),
            tmp_store=Path("unused-tmp.zarr"),
            get_template_fn=TEMPLATE_CONFIG.get_template,
            append_dim="init_time",
            all_data_vars=TEMPLATE_CONFIG.data_vars,
            reformat_job_name="test",
        )
    )

    (job,) = jobs
    assert isinstance(job, EcmwfIfsEnsForecast15DaySpatialRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.get_index("init_time")
    assert init_times[-1] == pd.Timestamp("2026-05-16T00:00")
    # 48h window over daily inits -> the last two init times.
    assert job.region == slice(len(init_times) - 2, len(init_times))


@pytest.mark.slow
def test_real_source_all_vars_resolve_and_decode(template_ds: xr.Dataset) -> None:
    """Guard against ECMWF layout/index drift: list the real bucket, parse a real
    index for every variable, and decode one message to check values + grid.

    ECMWF open data has ~4-day retention, so target a recent init.
    """
    init = (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)).tz_localize(
        None
    )
    lead = pd.Timedelta("6h")
    data_vars = TEMPLATE_CONFIG.data_vars
    coord = IfsEnsForecast15DaySpatialSourceFileCoord(
        init_time=init,
        lead_time=lead,
        ensemble_members=_PERTURBED,
        data_vars=data_vars,
    )
    job = make_job(template_ds)

    key = coord.get_url().removeprefix(_S3_LOCATION_PREFIX)
    available = job._discover_available({key: coord})
    if key not in available:
        pytest.skip(f"ECMWF open data for {init} not currently in the bucket")
    file_size = available[key]
    refs = job._file_refs(coord, file_size)

    # 17 vars x 50 perturbed members.
    assert len(refs) == 17 * 50
    for ref in refs:
        assert 0 <= ref.offset < file_size
        assert 0 < ref.length
        assert ref.offset + ref.length <= file_size
        assert ref.location == coord.get_url()

    t2m_ref = next(
        r
        for r in refs
        if r.data_var.name == "temperature_2m" and r.out_loc["ensemble_member"] == 1
    )
    https_url = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/" + key
    response = httpx.get(
        https_url,
        headers={
            "Range": f"bytes={t2m_ref.offset}-{t2m_ref.offset + t2m_ref.length - 1}"
        },
    )
    response.raise_for_status()
    data = parse_grib_array(response.content, 0)
    assert data.size == 721 * 1440
    assert 180 < np.nanmin(data) < 280 < np.nanmax(data) < 340  # plausible Kelvin

    # Virtual chunks serve the raw message grid; it must match the template exactly.
    lat, lon = parse_grib_message_metadata(response.content, 0).latlng()
    dim_coords = TEMPLATE_CONFIG.dimension_coordinates()
    np.testing.assert_allclose(np.asarray(lat), dim_coords["latitude"])
    np.testing.assert_allclose(np.asarray(lon), dim_coords["longitude"])
