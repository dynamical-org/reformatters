from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.google.weathernext2.forecast_virtual import (
    region_job as region_job_module,
)
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    AVAILABILITY_LOCATION_PREFIX,
    OUTPUT_CHUNK_LENGTH,
    PROXY_LOCATION_PREFIX,
    SOURCE_LOCATION_PREFIX,
    GoogleWeathernext2ForecastVirtualRegionJob,
    GoogleWeathernext2ForecastVirtualSourceFileCoord,
)
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    PRESSURE_LEVELS,
    GoogleWeathernext2DataVar,
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)

TEMPLATE_CONFIG = GoogleWeathernext2ForecastVirtualTemplateConfig()

_ANNUAL_INIT = pd.Timestamp("2022-01-02T12:00")
_OPERATIONAL_INIT = pd.Timestamp("2025-03-01T06:00")
_LEAD = pd.Timedelta("12h")
_ANNUAL_STORE = (
    f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0/zarr/2022_to_2023/predictions.zarr"
)
_OPERATIONAL_STORE = (
    f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0/zarr/2025_to_present/"
    "20250301_06hr_01_preds/predictions.zarr"
)


def get_var(path: str) -> GoogleWeathernext2DataVar:
    return next(var for var in TEMPLATE_CONFIG.data_vars if var.path == path)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2022-01-01T06:00"))


def make_job(
    template_ds: xr.DataTree,
    data_vars: Sequence[GoogleWeathernext2DataVar] | None = None,
    processing_mode: Literal["backfill", "update"] = "backfill",
    publication_cutoff: pd.Timestamp = pd.Timestamp.max,
) -> GoogleWeathernext2ForecastVirtualRegionJob:
    return GoogleWeathernext2ForecastVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars or TEMPLATE_CONFIG.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode=processing_mode,
        publication_cutoff=publication_cutoff,
    )


def make_coord(
    data_vars: Sequence[GoogleWeathernext2DataVar],
    init_time: pd.Timestamp = _OPERATIONAL_INIT,
    lead_time: pd.Timedelta = _LEAD,
) -> GoogleWeathernext2ForecastVirtualSourceFileCoord:
    return GoogleWeathernext2ForecastVirtualSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        data_vars=data_vars,
    )


def test_source_urls_and_success_markers() -> None:
    annual = make_coord([get_var("temperature_2m")], _ANNUAL_INIT)
    assert annual.get_url() == _ANNUAL_STORE
    assert annual.get_success_marker_url() == (
        f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0/zarr/2022_to_2023/success"
    )

    operational = make_coord([get_var("temperature_2m")])
    assert operational.get_url() == _OPERATIONAL_STORE
    assert operational.get_success_marker_url() == (
        f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0/zarr/2025_to_present/"
        "20250301_06hr_01_preds/success"
    )


def test_annual_root_chunk_and_first_middle_final_member_planes() -> None:
    var = get_var("temperature_2m")
    coord = make_coord([var], _ANNUAL_INIT)
    assert coord.chunk_key(var, 0, None) == "2m_temperature/6.0.1.0.0"
    assert coord.plane_index(var, 0, None) == 0
    assert coord.chunk_key(var, 31, None) == "2m_temperature/6.7.1.0.0"
    assert coord.plane_index(var, 31, None) == 3
    assert coord.chunk_key(var, 63, None) == "2m_temperature/6.15.1.0.0"
    assert coord.plane_index(var, 63, None) == 3


def test_annual_pressure_chunk_and_plane_include_member_and_level() -> None:
    var = get_var("pressure_level/temperature")
    coord = make_coord([var], _ANNUAL_INIT)
    assert coord.chunk_key(var, 0, 50) == "temperature/6.0.1.0.0.0"
    assert coord.plane_index(var, 0, 50) == 0
    assert coord.chunk_key(var, 31, 500) == "temperature/6.7.1.0.0.0"
    assert coord.plane_index(var, 31, 500) == 46
    assert coord.chunk_key(var, 63, 1000) == "temperature/6.15.1.0.0.0"
    assert coord.plane_index(var, 63, 1000) == 51


def test_operational_chunks_are_singleton_planes() -> None:
    root_var = get_var("temperature_2m")
    pressure_var = get_var("pressure_level/temperature")
    coord = make_coord([root_var, pressure_var])
    assert coord.chunk_key(root_var, 63, None) == "2m_temperature/63.1.0.0"
    assert coord.plane_index(root_var, 63, None) == 0
    assert coord.chunk_key(pressure_var, 63, 1000) == "temperature/63.1.12.0.0"
    assert coord.plane_index(pressure_var, 63, 1000) == 0


def test_generate_source_file_coords_enforces_strict_publication_cutoff(
    template_ds: xr.DataTree,
) -> None:
    init_time = pd.Timestamp("2025-03-01T00:00")
    cutoff = init_time + pd.Timedelta("12h")
    job = make_job(template_ds, publication_cutoff=cutoff)
    region_ds = (
        template_ds.to_dataset()
        .isel(init_time=slice(0, 1))
        .assign_coords(init_time=[init_time])
    )

    coords = job.generate_source_file_coords(region_ds, [get_var("temperature_2m")])

    assert [coord.lead_time for coord in coords] == [pd.Timedelta("6h")]


def test_annual_root_refs_are_fixed_length_proxy_planes(
    template_ds: xr.DataTree,
) -> None:
    var = get_var("temperature_2m")
    coord = make_coord([var], _ANNUAL_INIT)
    refs = make_job(template_ds, [var]).file_refs(coord, file_size=0)

    assert len(refs) == 64
    assert refs[0].location == (
        f"{PROXY_LOCATION_PREFIX}plane/0/weathernext_2_0_0/zarr/"
        "2022_to_2023/predictions.zarr/2m_temperature/6.0.1.0.0"
    )
    assert refs[31].location.startswith(f"{PROXY_LOCATION_PREFIX}plane/3/")
    assert refs[31].location.endswith("/2m_temperature/6.7.1.0.0")
    assert refs[63].location.startswith(f"{PROXY_LOCATION_PREFIX}plane/3/")
    assert refs[63].location.endswith("/2m_temperature/6.15.1.0.0")
    assert all((ref.offset, ref.length) == (0, OUTPUT_CHUNK_LENGTH) for ref in refs)
    assert all("generation=" not in ref.location for ref in refs)


def test_annual_pressure_refs_cover_all_members_and_levels(
    template_ds: xr.DataTree,
) -> None:
    var = get_var("pressure_level/temperature")
    coord = make_coord([var], _ANNUAL_INIT)
    refs = make_job(template_ds, [var]).file_refs(coord, file_size=0)

    assert len(refs) == 64 * len(PRESSURE_LEVELS)
    assert refs[0].location.startswith(f"{PROXY_LOCATION_PREFIX}plane/12/")
    assert refs[0].location.endswith("/temperature/6.0.1.0.0.0")
    assert refs[-1].location.startswith(f"{PROXY_LOCATION_PREFIX}plane/39/")
    assert refs[-1].location.endswith("/temperature/6.15.1.0.0.0")
    assert refs[0].out_loc["pressure_level"] == 1000
    assert refs[-1].out_loc["pressure_level"] == 50


def test_operational_refs_use_singleton_source_chunks(
    template_ds: xr.DataTree,
) -> None:
    var = get_var("temperature_2m")
    coord = make_coord([var])
    refs = make_job(template_ds, [var]).file_refs(coord, file_size=0)

    assert len(refs) == 64
    assert refs[63].location == (
        f"{PROXY_LOCATION_PREFIX}plane/0/weathernext_2_0_0/zarr/2025_to_present/"
        "20250301_06hr_01_preds/predictions.zarr/"
        "2m_temperature/63.1.0.0"
    )


def test_discover_available_gates_and_deduplicates_success_markers(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = make_coord([get_var("temperature_2m")])
    unavailable_lead = make_coord(
        [get_var("temperature_2m")], lead_time=pd.Timedelta("18h")
    )
    missing = make_coord(
        [get_var("temperature_2m")],
        init_time=pd.Timestamp("2025-03-01T12:00"),
    )
    annual = make_coord(
        [get_var("temperature_2m")],
        init_time=_ANNUAL_INIT,
        lead_time=pd.Timedelta("360h"),
    )
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)

    def response_for(url: str) -> Mock:
        if "2022_to_2023" in url:
            return Mock(
                status_code=200,
                headers={"X-WeatherNext-Available-Lead-Count": "60"},
            )
        if "20250301_06hr" in url:
            return Mock(
                status_code=200,
                headers={"X-WeatherNext-Available-Lead-Count": "2"},
            )
        return Mock(status_code=404)

    client.head.side_effect = response_for
    monkeypatch.setattr(region_job_module.httpx, "Client", lambda **kwargs: client)

    result = make_job(template_ds).discover_available(
        [published, unavailable_lead, missing, annual]
    )

    assert [coord for coord, _ in result] == [published, annual]
    assert client.head.call_count == 3
    assert all(
        call.args[0].startswith(AVAILABILITY_LOCATION_PREFIX)
        for call in client.head.call_args_list
    )


def test_backfill_process_virtual_refs_yields_four_inits_per_batch(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    var = get_var("temperature_2m")
    init_times = pd.date_range("2025-03-01T06:00", periods=5, freq="6h")
    coords = [
        make_coord([var], init_time, lead_time)
        for lead_time in (pd.Timedelta("6h"), pd.Timedelta("12h"))
        for init_time in init_times
    ]
    batch_template = TEMPLATE_CONFIG.get_template(init_times[-1] + pd.Timedelta("6h"))
    monkeypatch.setattr(
        GoogleWeathernext2ForecastVirtualRegionJob,
        "discover_available",
        lambda self, pending: [(coord, 0) for coord in pending],
    )
    monkeypatch.setattr(
        GoogleWeathernext2ForecastVirtualRegionJob,
        "file_refs",
        lambda self, coord, file_size: [Mock()],
    )

    batches = list(make_job(batch_template, [var]).process_virtual_refs(coords))

    assert [[coord for coord, _ in batch] for batch in batches] == [
        [
            coords[0],
            coords[5],
            coords[1],
            coords[6],
            coords[2],
            coords[7],
        ],
        [coords[3], coords[8], coords[4], coords[9]],
    ]


def test_update_process_virtual_refs_keeps_ready_inits_in_one_batch(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    var = get_var("temperature_2m")
    coords = [
        make_coord([var], pd.Timestamp("2025-03-01T00:00")),
        make_coord([var], pd.Timestamp("2025-03-01T06:00")),
    ]
    monkeypatch.setattr(
        GoogleWeathernext2ForecastVirtualRegionJob,
        "discover_available",
        lambda self, pending: [(coord, 0) for coord in pending],
    )
    monkeypatch.setattr(
        GoogleWeathernext2ForecastVirtualRegionJob,
        "file_refs",
        lambda self, coord, file_size: [Mock()],
    )

    batches = list(
        make_job(template_ds, [var], processing_mode="update").process_virtual_refs(
            coords
        )
    )

    assert [[coord for coord, _ in batch] for batch in batches] == [coords]


def test_operational_jobs_share_fire_time_and_48h_cutoff() -> None:
    fire_time = pd.Timestamp("2025-03-03T06:55")
    jobs, template_ds = (
        GoogleWeathernext2ForecastVirtualRegionJob.operational_update_jobs(
            primary_store=Mock(),
            tmp_store=Path("unused-tmp.zarr"),
            get_template_fn=TEMPLATE_CONFIG.get_template,
            append_dim="init_time",
            all_data_vars=TEMPLATE_CONFIG.data_vars,
            reformat_job_name="test",
            job_fire_time=fire_time,
        )
    )

    (job,) = jobs
    assert isinstance(job, GoogleWeathernext2ForecastVirtualRegionJob)
    assert job.processing_mode == "update"
    assert job.publication_cutoff == fire_time - pd.Timedelta("48h")
    init_times = template_ds.to_dataset().get_index("init_time")
    expected_start = init_times.searchsorted(fire_time - pd.Timedelta("18D"))
    assert job.region == slice(expected_start, len(init_times))
