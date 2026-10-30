from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import httpx
import icechunk
import pandas as pd
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from reformatters.common import template_utils
from reformatters.common.config_models import ROOT
from reformatters.google.weathernext2.forecast_historical_virtual.template_config import (
    GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig,
)
from reformatters.google.weathernext2.forecast_operational_virtual.template_config import (
    GoogleWeathernext2ForecastOperationalVirtualTemplateConfig,
)
from reformatters.google.weathernext2.forecast_virtual import (
    region_job as region_job_module,
)
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    GoogleWeathernext2ForecastHistoricalVirtualRegionJob,
    GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    GoogleWeathernext2ForecastVirtualSourceFileCoord,
)
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2DataVar,
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)
from reformatters.google.weathernext_virtual import listing as listing_module
from reformatters.google.weathernext_virtual.listing import (
    OBJECTS_LOCATION,
    PROXY_LOCATION_PREFIX,
)

HISTORICAL = GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig()
OPERATIONAL = GoogleWeathernext2ForecastOperationalVirtualTemplateConfig()


@pytest.fixture
def unrestricted_update_job() -> GoogleWeathernext2ForecastOperationalVirtualRegionJob:
    return GoogleWeathernext2ForecastOperationalVirtualRegionJob(
        tmp_store=Path("unused.zarr"),
        template_ds=OPERATIONAL.get_template(pd.Timestamp("2025-01-01T18:00")),
        data_vars=OPERATIONAL.data_vars,
        append_dim="init_time",
        region=slice(0, 3),
        reformat_job_name="test",
        processing_mode="update",
        poll_deadline=pd.Timestamp("2000-01-01"),
    )


def test_unrestricted_update_rejects_backlog_before_discovery(
    unrestricted_update_job: GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = unrestricted_update_job
    job = job.model_copy(
        update={
            "template_ds": OPERATIONAL.get_template(pd.Timestamp("2025-01-03")),
            "region": slice(0, 8),
        }
    )
    coords = job.source_file_coords()
    newest = max(coord.init_time for coord in coords)
    pending = [
        coord
        for coord in coords
        if coord.init_time == newest or coord.data_vars[0].group is not ROOT
    ]
    assert len(pending) == 3360
    discovery = Mock(side_effect=AssertionError("discovery must not run"))
    monkeypatch.setattr(type(job), "discover_available", discovery)
    with pytest.raises(
        AssertionError,
        match=r"2426880.*1320960.*backfill-kubernetes",
    ):
        list(job.process_virtual_refs(pending))
    discovery.assert_not_called()


def test_unrestricted_update_admits_steady_state(
    unrestricted_update_job: GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = unrestricted_update_job
    coords = job.source_file_coords()
    oldest = min(coord.init_time for coord in coords)
    first_lead = min(coord.lead_time for coord in coords)
    pending = [
        coord
        for coord in coords
        if coord.init_time != oldest or coord.lead_time == first_lead
    ]
    per_init = len(job.template_ds.to_dataset().lead_time) * len(job.data_vars)
    assert len(pending) == 2 * per_init + len(job.data_vars) == 1694
    discovery = Mock(return_value=[])
    monkeypatch.setattr(type(job), "discover_available", discovery)
    assert list(job.process_virtual_refs(pending)) == []
    discovery.assert_called_once_with(pending)


@pytest.mark.parametrize("members", [1, 64])
def test_unrestricted_update_reference_bound(
    unrestricted_update_job: GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    monkeypatch: pytest.MonkeyPatch,
    members: int,
) -> None:
    job = unrestricted_update_job.model_copy(
        update={
            "template_ds": OPERATIONAL.get_template(pd.Timestamp("2025-01-02")).isel(
                ensemble_member=slice(0, members)
            ),
            "region": slice(0, 4),
        }
    )
    pending = job.source_file_coords()
    first_init = min(coord.init_time for coord in pending)
    refs_per_init = sum(
        len(job._source_chunks(coord))
        for coord in pending
        if coord.init_time == first_init
    )
    bound = job.manifest_init_split * refs_per_init
    assert bound == 4 * 60 * members * (8 + 6 * 13)
    discovery = Mock(return_value=[])
    monkeypatch.setattr(type(job), "discover_available", discovery)
    assert list(job.process_virtual_refs(pending)) == []
    discovery.assert_called_once_with(pending)

    extra = next(coord for coord in pending if coord.data_vars[0].group is ROOT)
    assert len(job._source_chunks(extra)) == members
    discovery.reset_mock()
    with pytest.raises(
        AssertionError,
        match=rf"{bound + members}.*{bound}.*backfill-kubernetes",
    ):
        list(job.process_virtual_refs([*pending, extra]))
    discovery.assert_not_called()


def test_unrestricted_eligibility_does_not_compare_valid_time(
    unrestricted_update_job: GoogleWeathernext2ForecastOperationalVirtualRegionJob,
) -> None:
    job = unrestricted_update_job
    region = job._processing_region_ds().isel(init_time=slice(0, 1))
    region = region.assign_coords(init_time=[pd.Timestamp.max])
    coords = job.generate_source_file_coords(region, job.data_vars)
    assert len(coords) == 60 * len(job.data_vars)


def _var(
    config: GoogleWeathernext2ForecastVirtualTemplateConfig, path: str
) -> GoogleWeathernext2DataVar:
    return next(var for var in config.data_vars if var.path == path)


def _job(
    cls: type[
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob
        | GoogleWeathernext2ForecastOperationalVirtualRegionJob
    ],
    config: GoogleWeathernext2ForecastVirtualTemplateConfig,
    template: xr.DataTree,
    data_vars: Sequence[GoogleWeathernext2DataVar],
) -> (
    GoogleWeathernext2ForecastHistoricalVirtualRegionJob
    | GoogleWeathernext2ForecastOperationalVirtualRegionJob
):
    return cls(
        tmp_store=Path("unused.zarr"),
        template_ds=template,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


def _coord(
    config: GoogleWeathernext2ForecastVirtualTemplateConfig,
    data_vars: Sequence[GoogleWeathernext2DataVar],
    init_time: pd.Timestamp,
) -> GoogleWeathernext2ForecastVirtualSourceFileCoord:
    return GoogleWeathernext2ForecastVirtualSourceFileCoord(
        source_layout=config.source_layout,
        init_time=init_time,
        lead_time=pd.Timedelta("12h"),
        data_vars=data_vars,
    )


def _mock_native_listing(
    monkeypatch: pytest.MonkeyPatch,
    job: (
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob
        | GoogleWeathernext2ForecastOperationalVirtualRegionJob
    ),
    coord: GoogleWeathernext2ForecastVirtualSourceFileCoord,
) -> Mock:
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    chunks = job._source_chunks(coord)
    midpoint = len(chunks) // 2

    def get(url: str, *, params: dict[str, str] | None = None) -> Mock:
        assert url == OBJECTS_LOCATION
        assert params is not None
        assert params["prefix"].startswith("weathernext_2_0_0/zarr/")
        assert params["maxResults"] == "1000"
        if coord.source_layout == "operational":
            assert params["matchGlob"] == (
                f"{params['prefix']}{region_job_module._OPERATIONAL_MEMBER_GLOB}."
                f"{coord.lead_index}.*"
            )
            assert params["delimiter"] == "/"
        else:
            assert "matchGlob" not in params
            assert "delimiter" not in params
        page_token = params.get("pageToken")
        page = chunks[:midpoint] if page_token is None else chunks[midpoint:]
        payload: dict[str, object] = {
            "items": [
                {
                    "name": chunk.location.removeprefix(PROXY_LOCATION_PREFIX),
                    "size": str(len(chunk.location) + 1000),
                    "md5Hash": "AAAAAAAAAAAAAAAAAAAAAA==",
                }
                for chunk in page
            ]
        }
        if page_token is None:
            payload["nextPageToken"] = "next"
        response = Mock(status_code=200)
        response.json.return_value = payload
        return response

    client.get.side_effect = get

    def client_factory(**kwargs: object) -> Mock:
        assert kwargs["headers"] == {"User-Agent": "dynamical.org reformatters"}
        return client

    monkeypatch.setattr(listing_module.httpx, "Client", client_factory)
    return client


def test_historical_refs_are_whole_four_member_native_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    var = _var(HISTORICAL, "temperature_2m")
    template = HISTORICAL.get_template(pd.Timestamp("2022-01-03T00:00"))
    job = _job(
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob,
        HISTORICAL,
        template,
        [var],
    )
    coord = _coord(HISTORICAL, [var], pd.Timestamp("2022-01-02T12:00"))
    client = _mock_native_listing(monkeypatch, job, coord)

    [(available_coord, _)] = job.discover_available([coord])
    refs = job.file_refs(available_coord, 0)

    assert len(refs) == 16
    assert refs[0].out_loc["ensemble_member"] == 0
    assert refs[-1].out_loc["ensemble_member"] == 60
    assert refs[0].location.endswith(
        "/2022_to_2023/predictions.zarr/2m_temperature/6.0.1.0.0"
    )
    assert refs[-1].location.endswith(
        "/2022_to_2023/predictions.zarr/2m_temperature/6.15.1.0.0"
    )
    assert all(ref.offset == 0 for ref in refs)
    assert all(ref.length == len(ref.location) + 1000 for ref in refs)
    assert all(
        ref.etag_checksum == '"00000000000000000000000000000000"' for ref in refs
    )
    assert client.get.call_count == 2


def test_historical_pressure_ref_covers_all_native_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    var = _var(HISTORICAL, "pressure_level/temperature")
    template = HISTORICAL.get_template(pd.Timestamp("2022-01-03T00:00"))
    job = _job(
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob,
        HISTORICAL,
        template,
        [var],
    )
    coord = _coord(HISTORICAL, [var], pd.Timestamp("2022-01-02T12:00"))
    _mock_native_listing(monkeypatch, job, coord)

    [(available_coord, _)] = job.discover_available([coord])
    refs = job.file_refs(available_coord, 0)

    assert len(refs) == 16
    assert refs[0].out_loc["pressure_level"] == 50
    assert refs[0].location.endswith(
        "/2022_to_2023/predictions.zarr/temperature/6.0.1.0.0.0"
    )


def test_operational_refs_keep_singleton_member_and_level_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    var = _var(OPERATIONAL, "pressure_level/temperature")
    template = OPERATIONAL.get_template(pd.Timestamp("2025-03-01T12:00"))
    job = _job(
        GoogleWeathernext2ForecastOperationalVirtualRegionJob,
        OPERATIONAL,
        template,
        [var],
    )
    coord = _coord(OPERATIONAL, [var], pd.Timestamp("2025-03-01T06:00"))
    _mock_native_listing(monkeypatch, job, coord)

    [(available_coord, _)] = job.discover_available([coord])
    refs = job.file_refs(available_coord, 0)

    assert len(refs) == 64 * 13
    assert refs[0].location.endswith(
        "/20250301_06hr_01_preds/predictions.zarr/temperature/0.1.0.0.0"
    )
    assert refs[-1].location.endswith(
        "/20250301_06hr_01_preds/predictions.zarr/temperature/63.1.12.0.0"
    )


def test_source_file_coords_are_independent_per_variable() -> None:
    init_time = pd.Timestamp("2025-03-01T00:00")
    template = OPERATIONAL.get_template(init_time + pd.Timedelta("6h"))
    data_vars = [
        _var(OPERATIONAL, "temperature_2m"),
        _var(OPERATIONAL, "pressure_level/temperature"),
    ]
    job = _job(
        GoogleWeathernext2ForecastOperationalVirtualRegionJob,
        OPERATIONAL,
        template,
        data_vars,
    )

    region = template.to_dataset().sel(init_time=[init_time])

    coords = job.generate_source_file_coords(region, data_vars)

    assert len(coords) == 2 * 60
    assert all(len(coord.data_vars) == 1 for coord in coords)
    assert {coord.data_vars[0].path for coord in coords} == {
        "temperature_2m",
        "pressure_level/temperature",
    }


def test_historical_validation_job_uses_final_fixed_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_clock_read() -> pd.Timestamp:
        raise AssertionError("historical jobs must not read the clock")

    monkeypatch.setattr(region_job_module, "_utc_now", fail_if_clock_read)
    jobs, template = (
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob.operational_update_jobs(
            primary_store=MemoryStore(),
            tmp_store=Path("unused.zarr"),
            get_template_fn=HISTORICAL.get_template,
            append_dim="init_time",
            all_data_vars=HISTORICAL.data_vars,
            reformat_job_name="test",
            job_fire_time=pd.Timestamp("2026-01-01T00:00"),
        )
    )

    [job] = jobs
    assert isinstance(job, GoogleWeathernext2ForecastHistoricalVirtualRegionJob)
    assert template.to_dataset().get_index("init_time")[-1] == pd.Timestamp(
        "2024-12-31T18:00"
    )
    source_coords = job.source_file_coords()
    assert source_coords
    last_init = pd.Timestamp("2024-12-31T18:00")
    assert max(coord.init_time for coord in source_coords) == last_init
    var = HISTORICAL.data_vars[0]
    assert [
        coord.lead_time
        for coord in source_coords
        if coord.init_time == last_init and coord.data_vars == (var,)
    ] == list(HISTORICAL.dimension_coordinates()["lead_time"])


def test_direct_operational_update_uses_utc_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2025-03-20T12:00")
    monkeypatch.setattr(region_job_module, "_utc_now", lambda: now)

    _jobs, template = (
        GoogleWeathernext2ForecastOperationalVirtualRegionJob.operational_update_jobs(
            primary_store=MemoryStore(),
            tmp_store=Path("unused.zarr"),
            get_template_fn=OPERATIONAL.get_template,
            append_dim="init_time",
            all_data_vars=OPERATIONAL.data_vars,
            reformat_job_name="test",
        )
    )

    assert template.to_dataset().get_index("init_time")[-1] == pd.Timestamp(
        "2025-03-20T06:00"
    )


def test_object_listing_retries_transient_response() -> None:
    prefix = "weathernext_2_0_0/zarr/store/temperature/"
    request = httpx.Request("GET", OBJECTS_LOCATION)
    transient = httpx.Response(502, request=request)
    success = httpx.Response(
        200,
        request=request,
        json={
            "items": [
                {
                    "name": f"{prefix}0.1.0.0",
                    "size": "100",
                    "md5Hash": "AAAAAAAAAAAAAAAAAAAAAA==",
                }
            ]
        },
    )
    client = Mock()
    client.get.side_effect = [transient, success]

    objects = listing_module.list_objects(
        client,
        listing_module.ObjectListingQuery(prefix),
    )

    assert objects == {
        f"{PROXY_LOCATION_PREFIX}{prefix}0.1.0.0": listing_module.NativeObjectMetadata(
            size=100,
            etag_checksum='"00000000000000000000000000000000"',
        )
    }
    assert client.get.call_count == 2


@pytest.mark.parametrize("historical", [True, False])
@pytest.mark.parametrize("pressure", [True, False])
def test_native_refs_and_paginated_queries_golden(
    monkeypatch: pytest.MonkeyPatch, historical: bool, pressure: bool
) -> None:
    config = HISTORICAL if historical else OPERATIONAL
    cls = (
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob
        if historical
        else GoogleWeathernext2ForecastOperationalVirtualRegionJob
    )
    init = pd.Timestamp("2022-01-02T12:00" if historical else "2025-03-01T06:00")
    var = _var(config, "pressure_level/temperature" if pressure else "temperature_2m")
    job = _job(cls, config, config.get_template(init + pd.Timedelta("6h")), [var])
    coord = _coord(config, [var], init)
    client = _mock_native_listing(monkeypatch, job, coord)
    [(available, size)] = job.discover_available([coord])
    refs = job.file_refs(available, size)

    store = (
        "2022_to_2023/predictions.zarr"
        if historical
        else "2025_to_present/20250301_06hr_01_preds/predictions.zarr"
    )
    source_name = "temperature" if pressure else "2m_temperature"
    prefix = f"weathernext_2_0_0/zarr/{store}/{source_name}/"
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    expected = []
    for member in range(0, 64, 4 if historical else 1):
        for level_index in range(1 if historical or not pressure else 13):
            key = f"6.{member // 4}.1" if historical else f"{member}.1"
            if pressure:
                key += f".{level_index}"
            location = f"https://wn.dynamical.org/chunks/{prefix}{key}.0.0"
            out_loc = {
                "init_time": init,
                "ensemble_member": member,
                "lead_time": pd.Timedelta("12h"),
            }
            if pressure:
                out_loc["pressure_level"] = levels[level_index]
            expected.append(
                (
                    var,
                    location,
                    0,
                    len(location) + 1000,
                    '"00000000000000000000000000000000"',
                    out_loc,
                )
            )
    assert [
        (
            ref.data_var,
            ref.location,
            ref.offset,
            ref.length,
            ref.etag_checksum,
            ref.out_loc,
        )
        for ref in refs
    ] == expected
    params = {"prefix": prefix + "6." if historical else prefix, "maxResults": "1000"}
    if not historical:
        params |= {
            "matchGlob": prefix + "{" + ",".join(map(str, range(64))) + "}.1.*",
            "delimiter": "/",
        }
    assert [(call.args, call.kwargs) for call in client.get.call_args_list] == [
        (("https://wn.dynamical.org/objects",), {"params": params}),
        (
            ("https://wn.dynamical.org/objects",),
            {"params": params | {"pageToken": "next"}},
        ),
    ]


def test_historical_backfill_is_unrestricted(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Mock(side_effect=AssertionError("historical jobs must not read the clock"))
    monkeypatch.setattr(region_job_module, "_utc_now", clock)
    jobs = GoogleWeathernext2ForecastHistoricalVirtualRegionJob.get_jobs(
        tmp_store=Path("unused.zarr"),
        template_ds=HISTORICAL.get_template(pd.Timestamp("2022-01-01T06:00")),
        append_dim="init_time",
        all_data_vars=HISTORICAL.data_vars,
        reformat_job_name="test",
    )
    assert jobs
    assert not clock.called


def test_unrestricted_backfill_includes_all_leads_without_reading_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Mock(side_effect=AssertionError("unrestricted backfill read the clock"))
    monkeypatch.setattr(region_job_module, "_utc_now", clock)
    init_time = pd.Timestamp("2025-01-01T00:00")
    template = OPERATIONAL.get_template(init_time + pd.Timedelta("6h"))

    [job] = GoogleWeathernext2ForecastOperationalVirtualRegionJob.get_jobs(
        tmp_store=Path("unused.zarr"),
        template_ds=template,
        append_dim="init_time",
        all_data_vars=OPERATIONAL.data_vars,
        reformat_job_name="test",
        filter_variable_names=["temperature_2m"],
    )

    assert [coord.lead_time for coord in job.source_file_coords()] == list(
        OPERATIONAL.dimension_coordinates()["lead_time"]
    )
    clock.assert_not_called()


@pytest.mark.parametrize("hour", [1, 7, 13, 19])
def test_unrestricted_update_includes_newest_init_and_revisits_missing_steps(
    hour: int,
) -> None:
    fire_time = pd.Timestamp("2025-03-20") + pd.Timedelta(hours=hour, minutes=5)
    newest_init = (fire_time - pd.Timedelta("6h")).floor("6h")
    older_init = newest_init - pd.Timedelta("18h")
    var = _var(OPERATIONAL, "temperature_2m")
    jobs, template = (
        GoogleWeathernext2ForecastOperationalVirtualRegionJob.operational_update_jobs(
            primary_store=MemoryStore(),
            tmp_store=Path("unused.zarr"),
            get_template_fn=OPERATIONAL.get_template,
            append_dim="init_time",
            all_data_vars=[var],
            reformat_job_name="test",
            job_fire_time=fire_time,
        )
    )
    [job] = jobs
    assert isinstance(job, GoogleWeathernext2ForecastOperationalVirtualRegionJob)
    init_times = template.to_dataset().get_index("init_time")
    assert init_times[-1] == newest_init
    assert job.region == slice(len(init_times) - 4, len(init_times))
    coords = job.source_file_coords()
    expected_steps = {
        (init_time, lead_time)
        for init_time in init_times[job.region]
        for lead_time in OPERATIONAL.dimension_coordinates()["lead_time"]
    }
    assert {(coord.init_time, coord.lead_time) for coord in coords} == expected_steps

    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    template_utils.write_metadata(
        template.isel(init_time=slice(0, -1)),
        session.store,
        "w-",
        consolidated=False,
        skip_icechunk_commit=True,
    )
    present_step = (older_init, pd.Timedelta("6h"))
    present = next(
        coord for coord in coords if (coord.init_time, coord.lead_time) == present_step
    )
    chunk_key = job.chunk_key(job.representative_probe_loc(present, var), var)
    assert chunk_key is not None
    session.store.set_virtual_refs(
        var.path,
        [
            icechunk.VirtualChunkSpec(
                index=list(chunk_key),
                location="https://example.invalid/chunk",
                offset=0,
                length=1,
            )
        ],
        validate_containers=False,
    )
    session.commit("existing short lead")

    remaining = job.filter_already_present(coords, repo.readonly_session("main").store)
    missing_steps = {(coord.init_time, coord.lead_time) for coord in remaining}
    assert missing_steps == expected_steps - {present_step}
    assert (newest_init, pd.Timedelta("360h")) in missing_steps
    assert (older_init, pd.Timedelta("360h")) in missing_steps


@pytest.mark.parametrize(
    ("fire_time", "newest_init"),
    [
        (pd.Timestamp("2025-03-20T12:00"), pd.Timestamp("2025-03-20T06:00")),
        (pd.Timestamp("2025-03-20T11:59:59"), pd.Timestamp("2025-03-20T00:00")),
        (pd.Timestamp("2025-03-20T12:05"), pd.Timestamp("2025-03-20T06:00")),
        (pd.Timestamp("2025-03-20T17:59"), pd.Timestamp("2025-03-20T06:00")),
    ],
)
def test_unrestricted_update_extent_tracks_source_landing(
    fire_time: pd.Timestamp, newest_init: pd.Timestamp
) -> None:
    jobs, template = (
        GoogleWeathernext2ForecastOperationalVirtualRegionJob.operational_update_jobs(
            primary_store=MemoryStore(),
            tmp_store=Path("unused.zarr"),
            get_template_fn=OPERATIONAL.get_template,
            append_dim="init_time",
            all_data_vars=[_var(OPERATIONAL, "temperature_2m")],
            reformat_job_name="test",
            job_fire_time=fire_time,
        )
    )

    [job] = jobs
    assert isinstance(job, GoogleWeathernext2ForecastOperationalVirtualRegionJob)
    assert template.to_dataset().get_index("init_time")[-1] == newest_init
    assert {
        coord.lead_time
        for coord in job.source_file_coords()
        if coord.init_time == newest_init
    } == set(OPERATIONAL.dimension_coordinates()["lead_time"])
