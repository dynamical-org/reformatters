import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest
from pydantic import ValidationError
from zarr.storage import MemoryStore

from reformatters.common import template_utils
from reformatters.google.weathernext3.forecast_virtual import region_job as job_module
from reformatters.google.weathernext3.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext3ForecastVirtualDataset,
)
from reformatters.google.weathernext3.forecast_virtual.region_job import (
    GoogleWeathernext3ForecastVirtualRegionJob,
)
from reformatters.google.weathernext3.forecast_virtual.source import (
    parse_wn3_source_location,
)
from reformatters.google.weathernext3.forecast_virtual.template_config import (
    STATISTICS,
    VARIABLE_SPECS,
)
from reformatters.google.weathernext_virtual.holdback_audit import (
    audit_source_locations,
    run_audit,
)
from scripts.validation import availability
from scripts.validation.manifest_scan import scan_manifest
from tests.google.weathernext3.forecast_virtual.datasets_test import DATASETS, job_for


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_reference_time_survives_clock_boundary_and_retry(
    dataset: GoogleWeathernext3ForecastVirtualDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = pd.Timestamp("2026-01-02T00:59:59Z").tz_localize(None)
    clock = Mock(
        side_effect=[
            reference.tz_localize(None) + pd.Timedelta(hours=i) for i in range(1000)
        ]
    )
    monkeypatch.setattr(job_module, "utc_now", clock)

    original_init = dataset.region_job_class.__init__

    def advance_clock(
        self: GoogleWeathernext3ForecastVirtualRegionJob,
        **values: Any,  # noqa: ANN401
    ) -> None:
        job_module.utc_now()
        original_init(self, **values)

    monkeypatch.setattr(dataset.region_job_class, "__init__", advance_clock)
    config = dataset.template_config
    kwargs: dict[str, Any] = {
        "tmp_store": Path("unused"),
        "template_ds": config.get_template(pd.Timestamp("2026-01-02")),
        "append_dim": "init_time",
        "all_data_vars": config.data_vars,
        "filter_variable_names": [config.data_vars[0].name],
        "reformat_job_name": "fixed-launch",
        "reference_time": reference,
        "append_dim_end": pd.Timestamp("2026-01-02"),
    }
    jobs = dataset.region_job_class.get_jobs(**kwargs)
    retry = dataset.region_job_class.get_jobs(**kwargs)
    assert len(jobs) > 1
    assert clock.call_count == 2 * len(jobs)
    for job in [*jobs, *retry]:
        assert job.launch_scope is not None
        assert job.launch_scope == jobs[0].launch_scope
        assert job.launch_scope.append_dim_end == pd.Timestamp("2026-01-02")
        assert job.launch_scope.filter_variable_names == [config.data_vars[0].path]
        assert job.reference_time == reference.tz_localize(None)
        assert job.publication_cutoff == pd.Timestamp("2026-01-01T23:59:59")
        assert (
            job.commit_metadata()["publication_cutoff"] == "2026-01-01T23:59:59+00:00"
        )
    assert [job.source_file_coords() for job in jobs] == [
        job.source_file_coords() for job in retry
    ]
    kwargs["reference_time"] = None
    with pytest.raises(ValidationError, match="publication_cutoff"):
        dataset.region_job_class.get_jobs(**kwargs)


def test_exact_source_paths() -> None:
    job = job_for(DATASETS[0], "2026-01-01T01:00")
    coord = job.source_file_coords()[0]
    assert (
        coord.get_url()
        == "gs://weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr/2026_to_present/20260101_00hr_01_preds/predictions.zarr"
    )
    assert (
        coord.chunk_location(job.data_vars[0], "mean")
        == "https://wn.dynamical.org/chunks/weathernext_3_0_0_statistics/zarr/2026_to_present/20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/0/0"
    )
    (query,) = job._listing_queries(coord)
    assert (
        query.prefix
        == "weathernext_3_0_0_statistics/zarr/2026_to_present/20260101_00hr_01_preds/predictions.zarr/temperature_2m_"
    )


PREFIX = (
    "https://wn.dynamical.org/chunks/weathernext_3_0_0_statistics/zarr/2026_to_present/"
)
BASES = [spec[1] for spec in VARIABLE_SPECS] + [
    "station_head_temperature_2m",
    "station_head_dewpoint_temperature_2m",
]


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("statistic", STATISTICS)
@pytest.mark.parametrize(
    ("hour", "index"), [(0, 359), (6, 359), (12, 359), (18, 359), (1, 47), (23, 47)]
)
def test_parser_accepts_all_source_arrays(
    base: str, statistic: str, hour: int, index: int
) -> None:
    assert parse_wn3_source_location(
        f"{PREFIX}20260101_{hour:02}hr_01_preds/predictions.zarr/{base}_{statistic}/c/{index}/0/0"
    ) == (
        pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour),
        pd.Timedelta(hours=index + 1),
    )


@pytest.mark.parametrize(
    "suffix",
    [
        "20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/360/0/0",
        "20260101_01hr_01_preds/predictions.zarr/temperature_2m_mean/c/48/0/0",
        "20250101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/0/0",
        "20260230_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/0/0",
        "20260101_24hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/0/0",
        "20260101_0hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/0/0",
        "20260101_00hr_02_preds/predictions.zarr/temperature_2m_mean/c/0/0/0",
        "20260101_00hr_01_preds/prediction.zarr/temperature_2m_mean/c/0/0/0",
        "20260101_00hr_01_preds/predictions.zarr/unknown_mean/c/0/0/0",
        "20260101_00hr_01_preds/predictions.zarr/temperature_2m_p99/c/0/0/0",
        "20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/00/0/0",
        "20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/-1/0/0",
        "20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/1/0",
        "20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/0/0/",
        "20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/0/0?x=1",
        "20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/zarr.json",
    ],
)
def test_parser_rejects_invalid_source_keys(suffix: str) -> None:
    with pytest.raises((AssertionError, ValueError)):
        parse_wn3_source_location(PREFIX + suffix)


@pytest.mark.parametrize(
    "prefix",
    [
        "http://wn.dynamical.org/chunks/",
        "https://evil.example/chunks/",
        "gs://weathernext3_statistics_spatial/",
        "https://wn.dynamical.org/chunks/weathernext_2_0_0/zarr/2026_to_present/",
    ],
)
def test_parser_rejects_unexpected_prefix(prefix: str) -> None:
    with pytest.raises(AssertionError):
        parse_wn3_source_location(
            prefix
            + "20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/0/0"
        )


@pytest.mark.parametrize("future", [False, True])
def test_source_audit_catches_future_source_under_eligible_target(
    future: bool, tmp_path: Path
) -> None:
    job = job_for(DATASETS[0], "2026-01-01T01:00")
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    template_utils.write_metadata(
        job.template_ds.isel(init_time=slice(0, 1)),
        session.store,
        "w-",
        consolidated=False,
        skip_icechunk_commit=True,
    )
    session.store.set_virtual_ref(
        "temperature_2m/c/0/5/0/0/0",
        f"{PREFIX}20260101_00hr_01_preds/predictions.zarr/temperature_2m_p90/c/{int(future)}/0/0",
        offset=0,
        length=16,
        validate_container=False,
    )
    snapshot = session.commit(
        "independently planted source", metadata=job.commit_metadata()
    )
    target_audit = run_audit(repo, "fixture", None, snapshot, None, tmp_path / "target")
    assert target_audit.present_keys == 0
    if future:
        with pytest.raises(AssertionError, match="1 invalid source locations"):
            audit_source_locations(
                repo, snapshot, parse_wn3_source_location, tmp_path / "source"
            )
    else:
        assert (
            audit_source_locations(
                repo, snapshot, parse_wn3_source_location, tmp_path / "source"
            ).total_locations
            == 1
        )


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
@pytest.mark.parametrize("route", ["direct", "availability"])
def test_scan_defaults_probe_every_statistic_and_count_each_lead(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
    route: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = job_for(dataset, "2026-01-01T03:00")
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    template_utils.write_metadata(
        job.template_ds.isel(init_time=slice(0, 2)),
        session.store,
        "w-",
        consolidated=False,
        skip_icechunk_commit=True,
    )
    name = job.data_vars[0].path
    for lead in range(2):
        for statistic in range(6):
            if lead == 1 and statistic == 5:
                continue
            session.store.set_virtual_ref(
                f"{name}/c/0/{statistic}/{lead}/0/0",
                f"{PREFIX}20260101_00hr_01_preds/predictions.zarr/temperature_2m_{STATISTICS[statistic]}/c/{lead}/0/0",
                offset=0,
                length=16,
                validate_container=False,
            )
    snapshot = session.commit(
        "one complete, one missing p90, one missing lead",
        metadata=job.commit_metadata(),
    )
    start = pd.Timestamp("2026-01-01")
    end = start + 2 * dataset.template_config.append_dim_frequency
    store = repo.readonly_session(snapshot_id=snapshot).store
    if route == "direct":
        result = scan_manifest(
            dataset,
            store,
            start=start,
            end=end,
            variables=[name],
            snapshot_metadata=repo.lookup_snapshot(snapshot).metadata,
            probe_workers=2,
        )
        counts = result.file_availability
    else:
        ctx = Mock(
            validation_url="unused",
            variables=[name],
            checkpoint_dir=None,
            probe_workers=2,
            output_dir=tmp_path,
        )
        monkeypatch.setattr(
            availability,
            "resolve_scan_window",
            lambda ctx: (dataset, store, start, end),
        )
        monkeypatch.setattr(availability, "open_icechunk_repository", lambda url: repo)
        counts = availability.run_manifest_scan(ctx)
        assert (tmp_path / ctx.unavailable_timestamps_file).exists()
    expected = {start: (1, 3)}
    if dataset.template_config.horizon_hours == 48:
        expected[start + pd.Timedelta("1h")] = (0, 2)
    assert counts == expected


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_wn3_scan_requires_provenance(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
) -> None:
    assert dataset.region_job_class.requires_scan_provenance is True
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    with pytest.raises(AssertionError, match="require a recorded cutoff"):
        scan_manifest(
            dataset,
            repo.readonly_session("main").store,
            start=pd.Timestamp("2026-01-01"),
            end=pd.Timestamp("2026-01-02"),
            snapshot_metadata={},
        )


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_update_metadata_records_reference_and_launch_scope(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
) -> None:
    config = dataset.template_config
    fire = pd.Timestamp("2026-02-01T05:10:00Z").tz_localize(None)
    jobs, template = dataset.region_job_class.operational_update_jobs(
        MemoryStore(),
        Path("unused"),
        config.get_template,
        "init_time",
        config.data_vars,
        "scheduled-update",
        fire,
    )
    (job,) = jobs
    assert isinstance(job, GoogleWeathernext3ForecastVirtualRegionJob)
    cutoff = fire - pd.Timedelta("1h")
    end = (cutoff.floor("h") - pd.Timedelta("1h")).floor(
        config.append_dim_frequency
    ) + config.append_dim_frequency
    start = end - job.operational_update_window
    assert job.reference_time == fire
    assert job.publication_cutoff == cutoff
    assert job.launch_scope is not None
    assert job.launch_scope.append_dim_end == end
    assert job.launch_scope.filter_start == start
    assert job.launch_scope.filter_end == end
    assert job.launch_scope.filter_variable_names == [
        var.path for var in config.data_vars
    ]
    assert template.to_dataset().get_index("init_time")[job.region.start] == start
    assert (
        template.to_dataset().get_index("init_time")[job.region.stop - 1]
        == end - config.append_dim_frequency
    )
    metadata = job.commit_metadata()
    assert metadata["publication_cutoff"] == cutoff.tz_localize("UTC").isoformat()
    assert metadata["reference_time"] == fire.tz_localize("UTC").isoformat()
    scope = json.loads(metadata["launch_scope"])
    assert scope["append_dim_end"] == end.tz_localize("UTC").isoformat()
    assert scope["filter_start"] == start.tz_localize("UTC").isoformat()
    assert scope["filter_end"] == end.tz_localize("UTC").isoformat()
    assert scope["filter_variable_names"] == [var.path for var in config.data_vars]
