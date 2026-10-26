from pathlib import Path
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.google.weathernext3.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext3ForecastVirtualDataset,
)
from reformatters.google.weathernext3.forecast_virtual.template_config import (
    STATISTICS,
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
