from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import httpx
import icechunk
import pandas as pd
import pytest
import xarray as xr
import zarr
from zarr.storage import MemoryStore

from reformatters.common import template_utils
from reformatters.common.config_models import ROOT
from reformatters.common.iterating import get_worker_jobs
from reformatters.common.region_job import LaunchScope
from reformatters.common.storage import commit_if_icechunk
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
from reformatters.google.weathernext_virtual.holdback import PUBLICATION_HOLDBACK
from reformatters.google.weathernext_virtual.listing import (
    OBJECTS_LOCATION,
    PROXY_LOCATION_PREFIX,
)
from scripts.validation.manifest_scan import _scan_launch_scope

HISTORICAL = GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig()
OPERATIONAL = GoogleWeathernext2ForecastOperationalVirtualTemplateConfig()


@pytest.fixture
def delayed_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        GoogleWeathernext2ForecastOperationalVirtualRegionJob,
        "publication_holdback",
        PUBLICATION_HOLDBACK,
    )


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
        publication_cutoff=pd.Timestamp.max,
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
    assert job._step_coords(region, job.data_vars, publishable=False) == []


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
    publication_cutoff: pd.Timestamp = pd.Timestamp.max,
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
        publication_cutoff=publication_cutoff,
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
    monkeypatch.setattr(listing_module.httpx, "Client", lambda **kwargs: client)
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


def _lead_times_published_at(
    fire_time: pd.Timestamp, init_time: pd.Timestamp
) -> list[pd.Timedelta]:
    template = OPERATIONAL.get_template(init_time + pd.Timedelta("6h"))
    var = _var(OPERATIONAL, "temperature_2m")
    region = template.to_dataset().sel(init_time=[init_time])
    job = _job(
        GoogleWeathernext2ForecastOperationalVirtualRegionJob,
        OPERATIONAL,
        template,
        [var],
        publication_cutoff=fire_time - PUBLICATION_HOLDBACK,
    )
    return [coord.lead_time for coord in job.generate_source_file_coords(region, [var])]


def test_publication_holdback_is_one_hour_after_valid_time(
    delayed_publication: None,
) -> None:
    assert (
        GoogleWeathernext2ForecastOperationalVirtualRegionJob.publication_holdback
        == pd.Timedelta("1h")
    )
    init_time = pd.Timestamp("2025-03-01T00:00")
    valid_time = init_time + pd.Timedelta("12h")
    one_second = pd.Timedelta("1s")

    assert _lead_times_published_at(valid_time + pd.Timedelta("1h"), init_time) == [
        pd.Timedelta("6h"),
        pd.Timedelta("12h"),
    ]
    assert _lead_times_published_at(
        valid_time + pd.Timedelta("1h") - one_second, init_time
    ) == [pd.Timedelta("6h")]
    assert _lead_times_published_at(
        valid_time + pd.Timedelta("1h") + one_second, init_time
    ) == [pd.Timedelta("6h"), pd.Timedelta("12h")]


def test_publication_holdback_ignores_initialization_age(
    delayed_publication: None,
) -> None:
    init_time = pd.Timestamp("2025-03-01T00:00")
    all_lead_times = list(OPERATIONAL.dimension_coordinates()["lead_time"])

    assert (
        _lead_times_published_at(init_time + pd.Timedelta("48h"), init_time)
        == all_lead_times[:7]
    )
    assert _lead_times_published_at(init_time + pd.Timedelta("6h"), init_time) == []
    assert _lead_times_published_at(
        init_time + pd.Timedelta("6h") + PUBLICATION_HOLDBACK, init_time
    ) == [pd.Timedelta("6h")]
    assert (
        _lead_times_published_at(
            init_time + pd.Timedelta("360h") + PUBLICATION_HOLDBACK, init_time
        )
        == all_lead_times
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


def test_backfill_jobs_apply_the_holdback_to_the_current_clock(
    delayed_publication: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_time = pd.Timestamp("2025-01-01T00:00")
    now = init_time + pd.Timedelta("18h") + PUBLICATION_HOLDBACK
    clock = Mock(side_effect=[now, now + pd.Timedelta("1h")])
    monkeypatch.setattr(region_job_module, "_utc_now", clock)
    template = OPERATIONAL.get_template(init_time + pd.Timedelta("6h"))

    [job] = GoogleWeathernext2ForecastOperationalVirtualRegionJob.get_jobs(
        tmp_store=Path("unused.zarr"),
        template_ds=template,
        append_dim="init_time",
        all_data_vars=OPERATIONAL.data_vars,
        reformat_job_name="test",
    )

    assert clock.call_count == 1
    assert job.publication_cutoff == init_time + pd.Timedelta("18h")
    region = template.to_dataset().isel(init_time=job.region)
    coords = job.generate_source_file_coords(
        region,
        [_var(OPERATIONAL, "temperature_2m")],
    )
    assert [coord.lead_time for coord in coords] == [
        pd.Timedelta("6h"),
        pd.Timedelta("12h"),
        pd.Timedelta("18h"),
    ]


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


def test_operational_update_publishes_diagonal_window(
    delayed_publication: None,
) -> None:
    fire_time = pd.Timestamp("2025-03-20T13:05")
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
    cutoff = fire_time - PUBLICATION_HOLDBACK
    assert job.publication_cutoff == cutoff
    assert template.to_dataset().get_index("init_time")[-1] == pd.Timestamp(
        "2025-03-20T06:00"
    )

    coords = job.source_file_coords()
    valid_times = {coord.init_time + coord.lead_time for coord in coords}
    assert max(valid_times) == pd.Timestamp("2025-03-20T12:00")
    init_times = template.to_dataset().get_index("init_time")
    window_start = init_times[-1] + pd.Timedelta("6h") - pd.Timedelta("17D")
    expected = {
        (init_time, lead_time)
        for init_time in init_times[init_times >= window_start]
        for lead_time in OPERATIONAL.dimension_coordinates()["lead_time"]
        if init_time + lead_time <= cutoff
    }
    assert {(coord.init_time, coord.lead_time) for coord in coords} == expected
    oldest_init = min(coord.init_time for coord in coords)
    newest_init = max(coord.init_time for coord in coords)
    assert oldest_init < cutoff - pd.Timedelta("360h")
    assert [
        coord.lead_time for coord in coords if coord.init_time == oldest_init
    ] == list(OPERATIONAL.dimension_coordinates()["lead_time"])
    assert [coord.lead_time for coord in coords if coord.init_time == newest_init] == [
        pd.Timedelta("6h")
    ]


@pytest.mark.parametrize(
    ("fire_time", "newest_init"),
    [
        (pd.Timestamp("2025-03-20T13:00"), pd.Timestamp("2025-03-20T06:00")),
        (pd.Timestamp("2025-03-20T12:59:59"), pd.Timestamp("2025-03-20T00:00")),
        (pd.Timestamp("2025-03-20T13:05"), pd.Timestamp("2025-03-20T06:00")),
        (pd.Timestamp("2025-03-20T18:59"), pd.Timestamp("2025-03-20T06:00")),
    ],
)
def test_operational_update_template_ends_at_newest_publishable_init(
    delayed_publication: None, fire_time: pd.Timestamp, newest_init: pd.Timestamp
) -> None:
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

    assert template.to_dataset().get_index("init_time")[-1] == newest_init
    [job] = jobs
    assert isinstance(job, GoogleWeathernext2ForecastOperationalVirtualRegionJob)
    coords = job.source_file_coords()
    assert max(coord.init_time for coord in coords) == newest_init
    assert [coord.lead_time for coord in coords if coord.init_time == newest_init] == [
        pd.Timedelta("6h")
    ]


def test_direct_operational_update_uses_utc_clock(
    delayed_publication: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2025-03-20T12:00")
    monkeypatch.setattr(region_job_module, "_utc_now", lambda: now)

    jobs, _template = (
        GoogleWeathernext2ForecastOperationalVirtualRegionJob.operational_update_jobs(
            primary_store=MemoryStore(),
            tmp_store=Path("unused.zarr"),
            get_template_fn=OPERATIONAL.get_template,
            append_dim="init_time",
            all_data_vars=OPERATIONAL.data_vars,
            reformat_job_name="test",
        )
    )

    [job] = jobs
    assert isinstance(job, GoogleWeathernext2ForecastOperationalVirtualRegionJob)
    assert job.publication_cutoff == now - PUBLICATION_HOLDBACK


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


def test_backfill_process_cutoffs_preserve_partitions_and_eligibility(
    delayed_publication: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = pd.Timestamp("2025-01-02T00:59:59")
    after = pd.Timestamp("2025-01-02T01:00:01")
    clock = Mock(side_effect=[before, after])
    monkeypatch.setattr(region_job_module, "_utc_now", clock)
    template = OPERATIONAL.get_template(pd.Timestamp("2025-01-03"))
    job_sets = [
        GoogleWeathernext2ForecastOperationalVirtualRegionJob.get_jobs(
            tmp_store=Path("unused.zarr"),
            template_ds=template,
            append_dim="init_time",
            all_data_vars=OPERATIONAL.data_vars,
            reformat_job_name="test",
        )
        for _ in range(2)
    ]

    def partitions(
        jobs: Sequence[GoogleWeathernext2ForecastOperationalVirtualRegionJob],
    ) -> list[tuple[slice, tuple[str, ...]]]:
        return [(job.region, tuple(var.path for var in job.data_vars)) for job in jobs]

    assert clock.call_count == 2
    assert partitions(job_sets[0]) == partitions(job_sets[1])
    for worker_index in range(2):
        assignments = [
            get_worker_jobs(jobs, worker_index, 2, worker_assignment="contiguous")
            for jobs in job_sets
        ]
        assert partitions(assignments[0]) == partitions(assignments[1])

    coord_sets = []
    for now, jobs in zip((before, after), job_sets, strict=True):
        cutoff = now - PUBLICATION_HOLDBACK
        assert {job.publication_cutoff for job in jobs} == {cutoff}
        coords = {
            (coord.init_time, coord.lead_time, coord.data_vars[0].path)
            for job in jobs
            for coord in job.source_file_coords()
        }
        assert all(init + lead <= cutoff for init, lead, _ in coords)
        coord_sets.append(coords)
    assert coord_sets[0] < coord_sets[1]
    assert all(
        before - PUBLICATION_HOLDBACK < init + lead <= after - PUBLICATION_HOLDBACK
        for init, lead, _ in coord_sets[1] - coord_sets[0]
    )


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


@pytest.mark.parametrize(
    ("fire", "start", "end"),
    [
        ("2025-03-20T12:59:59", "2025-03-03T06:00", "2025-03-20T06:00"),
        ("2025-03-20T13:00", "2025-03-03T12:00", "2025-03-20T12:00"),
        ("2025-03-20T19:05", "2025-03-03T18:00", "2025-03-20T18:00"),
    ],
)
@pytest.mark.usefixtures("delayed_publication")
def test_operational_window_golden(fire: str, start: str, end: str) -> None:
    jobs, template = (
        GoogleWeathernext2ForecastOperationalVirtualRegionJob.operational_update_jobs(
            primary_store=MemoryStore(),
            tmp_store=Path("unused.zarr"),
            get_template_fn=OPERATIONAL.get_template,
            append_dim="init_time",
            all_data_vars=OPERATIONAL.data_vars,
            reformat_job_name="test",
            job_fire_time=pd.Timestamp(fire),
        )
    )
    [job] = jobs
    assert isinstance(job, GoogleWeathernext2ForecastOperationalVirtualRegionJob)
    inits = template.to_dataset().get_index("init_time")[job.region]
    assert inits[0] == pd.Timestamp(start)
    assert inits[-1] + pd.Timedelta("6h") == pd.Timestamp(end)
    assert job.publication_cutoff == pd.Timestamp(fire) - pd.Timedelta("1h")
    metadata = job.commit_metadata()
    scope = LaunchScope.model_validate_json(metadata["launch_scope"])
    assert scope.filter_start == pd.Timestamp(start).tz_localize("UTC")
    assert (
        scope.filter_end == scope.append_dim_end == pd.Timestamp(end).tz_localize("UTC")
    )
    assert scope.filter_variable_names == [var.path for var in OPERATIONAL.data_vars]
    assert (
        metadata["reference_time"] == pd.Timestamp(fire).tz_localize("UTC").isoformat()
    )


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
    assert all(job.publication_cutoff == pd.Timestamp.max for job in jobs)
    assert all(not job.held_back_source_file_coords() for job in jobs)
    assert not clock.called


@pytest.mark.usefixtures("delayed_publication")
def test_launch_cutoff_survives_reversed_commits_and_hour_boundary_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    reference = pd.Timestamp("2025-01-02T00:59:59")
    clock = Mock(side_effect=AssertionError("Worker must reuse launch time"))
    monkeypatch.setattr(region_job_module, "_utc_now", clock)
    template = OPERATIONAL.get_template(pd.Timestamp("2025-01-03"))
    job_sets = [
        GoogleWeathernext2ForecastOperationalVirtualRegionJob.get_jobs(
            tmp_store=Path("unused.zarr"),
            template_ds=template,
            append_dim="init_time",
            all_data_vars=OPERATIONAL.data_vars,
            reformat_job_name="fixed-launch",
            reference_time=reference,
        )
        for _ in range(3)
    ]
    assert not clock.called
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    initial = repo.writable_session("main")
    zarr.open_group(initial.store, mode="w")
    initial.commit("setup", metadata=job_sets[0][0].commit_metadata())
    workers = [repo.writable_session("main") for _ in range(2)]
    for i, worker in enumerate(workers):
        zarr.open_group(worker.store, mode="a").create_array(
            f"worker-{i}", shape=(1,), dtype="float32"
        )
    for worker, job in reversed(
        list(zip(workers, (job_sets[0][0], job_sets[1][0]), strict=True))
    ):
        commit_if_icechunk("worker", worker.store, [], metadata=job.commit_metadata())
    retry = repo.writable_session("main")
    zarr.open_group(retry.store, mode="a").attrs["retry-after-01:00"] = True
    commit_if_icechunk(
        "retry", retry.store, [], metadata=job_sets[2][0].commit_metadata()
    )
    snapshots = list(repo.ancestry(branch="main"))[:-1]
    assert len(snapshots) == 4
    assert all(
        {
            key: snapshot.metadata.get(key)
            for key in ("publication_cutoff", "reformat_job_name")
        }
        == {
            "publication_cutoff": "2025-01-01T23:59:59+00:00",
            "reformat_job_name": "fixed-launch",
        }
        for snapshot in snapshots
    )
    coord_sets = [
        {
            (coord.init_time, coord.lead_time, coord.data_vars[0].path)
            for job in jobs
            for coord in job.source_file_coords()
        }
        for jobs in job_sets
    ]
    assert coord_sets[0] == coord_sets[1] == coord_sets[2]


def test_filtered_repair_records_launch_scope() -> None:

    end = pd.Timestamp("2025-01-03")
    start = pd.Timestamp("2025-01-01T06:00")
    filter_end = pd.Timestamp("2025-01-02T18:00")
    contains = [start, pd.Timestamp("2025-01-02T12:00")]
    variables = [OPERATIONAL.data_vars[0].path]
    jobs = GoogleWeathernext2ForecastOperationalVirtualRegionJob.get_jobs(
        tmp_store=Path("unused.zarr"),
        template_ds=OPERATIONAL.get_template(end),
        append_dim="init_time",
        all_data_vars=OPERATIONAL.data_vars,
        reformat_job_name="repair",
        reference_time=pd.Timestamp("2025-01-05"),
        append_dim_end=end,
        filter_start=start,
        filter_end=filter_end,
        filter_contains=contains,
        filter_variable_names=variables,
    )
    assert jobs
    metadata = jobs[0].commit_metadata()
    assert all(job.commit_metadata() == metadata for job in jobs)
    assert all(isinstance(value, str) for value in metadata.values())
    scope = LaunchScope.model_validate_json(metadata["launch_scope"])
    assert scope.append_dim_end == end.tz_localize("UTC")
    assert _scan_launch_scope(scope, OPERATIONAL.data_vars, None, None, None) == (
        start,
        filter_end,
        variables,
        contains,
    )


def test_historical_metadata_does_not_record_max_cutoff() -> None:
    job = _job(
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob,
        HISTORICAL,
        HISTORICAL.get_template(pd.Timestamp("2022-01-01T06:00")),
        HISTORICAL.data_vars,
    )
    metadata = job.commit_metadata()
    assert metadata["publication_policy"] == "unrestricted"
    assert "publication_cutoff" not in metadata
    assert "reference_time" not in metadata


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

    assert job.publication_holdback is None
    assert job.publication_cutoff == pd.Timestamp.max
    assert [coord.lead_time for coord in job.source_file_coords()] == list(
        OPERATIONAL.dimension_coordinates()["lead_time"]
    )
    assert not job.held_back_source_file_coords()
    clock.assert_not_called()


@pytest.mark.parametrize("hour", [1, 7, 13, 19])
def test_unrestricted_update_includes_newest_init_and_revisits_withheld_steps(
    hour: int,
) -> None:
    fire_time = pd.Timestamp("2025-03-20") + pd.Timedelta(hours=hour, minutes=5)
    newest_init = (fire_time - pd.Timedelta("7h")).floor("6h")
    older_init = newest_init - pd.Timedelta("2D")
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
    assert job.publication_cutoff == pd.Timestamp.max
    init_times = template.to_dataset().get_index("init_time")
    assert init_times[-1] == newest_init
    assert job.region == slice(
        int(
            init_times.searchsorted(
                newest_init + pd.Timedelta("6h") - pd.Timedelta("17D")
            )
        ),
        len(init_times),
    )
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
    assert older_init + pd.Timedelta("360h") > fire_time - PUBLICATION_HOLDBACK


@pytest.mark.parametrize(
    ("fire_time", "newest_init"),
    [
        (pd.Timestamp("2025-03-20T13:00"), pd.Timestamp("2025-03-20T06:00")),
        (pd.Timestamp("2025-03-20T12:59:59"), pd.Timestamp("2025-03-20T00:00")),
        (pd.Timestamp("2025-03-20T13:05"), pd.Timestamp("2025-03-20T06:00")),
        (pd.Timestamp("2025-03-20T18:59"), pd.Timestamp("2025-03-20T06:00")),
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
    assert job.publication_cutoff == pd.Timestamp.max
    assert template.to_dataset().get_index("init_time")[-1] == newest_init
    assert {
        coord.lead_time
        for coord in job.source_file_coords()
        if coord.init_time == newest_init
    } == set(OPERATIONAL.dimension_coordinates()["lead_time"])
