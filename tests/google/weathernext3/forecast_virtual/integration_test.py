from pathlib import Path
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.google.weathernext3.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext3ForecastVirtualDataset,
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
        "independently planted source",
        metadata={
            "publication_cutoff": job.publication_cutoff.tz_localize("UTC").isoformat()
        },
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
    snapshot = session.commit("one complete, one missing p90, one missing lead")
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
        counts = availability.run_manifest_scan(ctx)
        assert (tmp_path / ctx.unavailable_timestamps_file).exists()
    horizon = dataset.template_config.horizon_hours
    second = start + dataset.template_config.append_dim_frequency
    expected = {start: (1, horizon), second: (0, horizon)}
    assert counts == expected
