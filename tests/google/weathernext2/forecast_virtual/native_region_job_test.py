from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
import xarray as xr
from zarr.storage import MemoryStore

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
    OBJECTS_LOCATION,
    PROXY_LOCATION_PREFIX,
    GoogleWeathernext2ForecastHistoricalVirtualRegionJob,
    GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    GoogleWeathernext2ForecastVirtualSourceFileCoord,
)
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2DataVar,
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)

HISTORICAL = GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig()
OPERATIONAL = GoogleWeathernext2ForecastOperationalVirtualTemplateConfig()


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
    monkeypatch.setattr(region_job_module.httpx, "Client", lambda **kwargs: client)
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


def test_operational_generation_enforces_strict_init_time_cutoff() -> None:
    init_time = pd.Timestamp("2025-03-01T00:00")
    template = OPERATIONAL.get_template(init_time + pd.Timedelta("6h"))
    var = _var(OPERATIONAL, "temperature_2m")
    region = template.to_dataset().sel(init_time=[init_time])
    all_lead_times = list(OPERATIONAL.dimension_coordinates()["lead_time"])

    def coords_with_cutoff(cutoff: pd.Timestamp) -> list[pd.Timedelta]:
        job = _job(
            GoogleWeathernext2ForecastOperationalVirtualRegionJob,
            OPERATIONAL,
            template,
            [var],
            publication_cutoff=cutoff,
        )
        return [
            coord.lead_time for coord in job.generate_source_file_coords(region, [var])
        ]

    # An initialization before the cutoff publishes every lead time, however far ahead
    # it forecasts; one at the cutoff publishes nothing.
    assert (
        coords_with_cutoff(init_time + OPERATIONAL.append_dim_frequency)
        == all_lead_times
    )
    assert coords_with_cutoff(init_time) == []


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


def test_operational_backfill_jobs_share_strict_init_time_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_time = pd.Timestamp("2025-01-01T00:00")
    cutoff = init_time + pd.Timedelta("12h")
    monkeypatch.setattr(
        region_job_module,
        "_current_publication_cutoff",
        lambda: cutoff,
    )
    template = OPERATIONAL.get_template(init_time + pd.Timedelta("6h"))

    [job] = GoogleWeathernext2ForecastOperationalVirtualRegionJob.get_jobs(
        tmp_store=Path("unused.zarr"),
        template_ds=template,
        append_dim="init_time",
        all_data_vars=OPERATIONAL.data_vars,
        reformat_job_name="test",
    )

    assert job.publication_cutoff == cutoff
    region = template.to_dataset().isel(init_time=job.region)
    coords = job.generate_source_file_coords(
        region,
        [_var(OPERATIONAL, "temperature_2m")],
    )
    assert [coord.lead_time for coord in coords] == list(
        OPERATIONAL.dimension_coordinates()["lead_time"]
    )


def test_historical_validation_job_uses_final_fixed_window() -> None:
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
    assert max(coord.init_time for coord in source_coords) == pd.Timestamp(
        "2024-12-31T18:00"
    )


def test_direct_operational_update_uses_utc_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2025-03-20T12:00")
    monkeypatch.setattr(region_job_module, "_utc_now", lambda: now)

    jobs, template = (
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
    assert job.publication_cutoff == now - pd.Timedelta("48h")
    # The newest initialization kept is the newest one outside the publication lag.
    last_init = template.to_dataset().get_index("init_time")[-1]
    assert last_init < job.publication_cutoff
    assert last_init + OPERATIONAL.append_dim_frequency >= job.publication_cutoff


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

    objects = region_job_module._list_objects(
        client,
        region_job_module.ObjectListingQuery(prefix),
    )

    assert objects == {
        f"{PROXY_LOCATION_PREFIX}{prefix}0.1.0.0": region_job_module.NativeObjectMetadata(
            size=100,
            etag_checksum='"00000000000000000000000000000000"',
        )
    }
    assert client.get.call_count == 2
