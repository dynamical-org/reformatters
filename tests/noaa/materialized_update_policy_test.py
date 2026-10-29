from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.region_job import SourceFileResult, SourceFileStatus
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.gefs.analysis.dynamical_dataset import GefsAnalysisDataset
from reformatters.noaa.gefs.analysis.region_job import GefsAnalysisSourceFileCoord
from reformatters.noaa.gefs.forecast_35_day.dynamical_dataset import (
    GefsForecast35DayDataset,
)
from reformatters.noaa.gefs.forecast_35_day.region_job import GefsForecast35DayRegionJob
from reformatters.noaa.gefs.gefs_config_models import GEFS_PRE_EXTENSION_MAX
from reformatters.noaa.gfs.analysis.dynamical_dataset import NoaaGfsAnalysisDataset
from reformatters.noaa.gfs.forecast.dynamical_dataset import NoaaGfsForecastDataset
from reformatters.noaa.gfs.region_job import NoaaGfsSourceFileCoord
from reformatters.noaa.hrrr.analysis.dynamical_dataset import NoaaHrrrAnalysisDataset
from reformatters.noaa.mrms.conus_analysis_hourly.dynamical_dataset import (
    NoaaMrmsConusAnalysisHourlyDataset,
)
from tests.common.current_data_deadline_test import _check_at_time

type NoaaMaterializedDataset = (
    NoaaGfsForecastDataset
    | NoaaGfsAnalysisDataset
    | GefsAnalysisDataset
    | GefsForecast35DayDataset
    | NoaaHrrrAnalysisDataset
    | NoaaMrmsConusAnalysisHourlyDataset
)

VALUES = (
    (NoaaGfsForecastDataset, "44 5,11,17,23 * * *", "5h44m", 6),
    (NoaaGfsAnalysisDataset, "58 3,9,15,21 * * *", "3h58m", 6),
    (GefsAnalysisDataset, "3 4,10,16,22 * * *", "4h03m", 6),
    (GefsForecast35DayDataset, "15 7 * * *", "7h15m", 24),
    (NoaaHrrrAnalysisDataset, "10 1,4,7,10,13,16,19,22 * * *", "3h10m", 3),
    (NoaaMrmsConusAnalysisHourlyDataset, "3 * * * *", "5m", 1),
)


def _dataset(
    dataset_class: type[NoaaMaterializedDataset], tmp_path: Path
) -> DynamicalDataset:
    dataset = dataset_class(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ZARR3
        )
    )
    return dataset.model_copy(
        update={
            "template_config": dataset.template_config.model_copy(
                update={"append_dim_start": pd.Timestamp("2026-10-25")}
            )
        }
    )


@pytest.mark.parametrize(("dataset_class", "schedule", "max_delay", "hours"), VALUES)
def test_materialized_noaa_lifecycle_values(
    dataset_class: type[NoaaMaterializedDataset],
    schedule: str,
    max_delay: str,
    hours: int,
    tmp_path: Path,
) -> None:
    dataset = _dataset(dataset_class, tmp_path)
    (cron,) = dataset.operational_kubernetes_resources("test")
    assert cron.schedule == schedule
    assert cron.concurrency_policy == "Allow"
    spec = cron.as_kubernetes_object()["spec"]["jobTemplate"]["spec"]
    assert spec["activeDeadlineSeconds"] == min(hours, 6) * 3600
    assert spec["template"]["spec"]["activeDeadlineSeconds"] == min(hours, 6) * 3600
    (currency,) = [
        check
        for check in dataset.validators()
        if isinstance(check, validation.CheckCurrentData)
    ]
    assert currency.max_delay == pd.Timedelta(max_delay)


@pytest.mark.parametrize("dataset_class", [row[0] for row in VALUES])
def test_reference_keeps_all_noaa_builder_windows_stable(
    dataset_class: type[NoaaMaterializedDataset],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(dataset_class, tmp_path)
    append_dim = dataset.template_config.append_dim
    monkeypatch.setattr(
        xr,
        "open_zarr",
        Mock(
            return_value=xr.Dataset(
                coords={append_dim: [pd.Timestamp("2026-10-25T12:00")]}
            )
        ),
    )
    reference = pd.Timestamp("2026-10-26T10:26")
    windows: list[tuple[tuple[tuple[int, int], ...], xr.DataTree]] = []
    for now in ("2026-10-26T23:59:50", "2026-10-27T00:00:10"):
        monkeypatch.setattr(
            pd.Timestamp,
            "now",
            lambda *args, current=now, **kwargs: pd.Timestamp(current),
        )
        jobs, template = dataset._operational_update_jobs(
            "job", tmp_path, job_reference_time=reference
        )
        windows.append(
            (tuple((job.region.start, job.region.stop) for job in jobs), template)
        )
    assert windows[0][0] == windows[1][0]
    assert windows[0][1].coords[append_dim].equals(windows[1][1].coords[append_dim])
    assert dict(windows[0][1].sizes) == dict(windows[1][1].sizes)


@pytest.mark.parametrize("hour", range(1, 24, 3))
def test_hrrr_currency_requires_the_first_covered_anchor_at_each_fire(
    hour: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(NoaaHrrrAnalysisDataset, tmp_path)
    (cron,) = dataset.operational_kubernetes_resources("test")
    fire = pd.Timestamp("2026-10-26") + timedelta(hours=hour, minutes=10)
    assert cron.previous_fire_time(fire) == fire
    earliest = fire - pd.Timedelta("3h10m")
    previous = earliest - timedelta(hours=1)
    dataset_id = dataset.dataset_id
    assert _check_at_time(
        dataset_id, fire - timedelta(seconds=1), previous, monkeypatch
    ).passed
    assert not _check_at_time(dataset_id, fire, previous, monkeypatch).passed
    assert _check_at_time(dataset_id, fire, earliest, monkeypatch).passed


@pytest.mark.parametrize(
    ("dataset_class", "fire_offset", "last_complete"),
    [(NoaaGfsAnalysisDataset, "3h58m", "5h"), (GefsAnalysisDataset, "4h03m", "3h")],
)
def test_analysis_boundary_input_publishes_complete_cycle_and_trims_boundary(
    dataset_class: type[NoaaMaterializedDataset],
    fire_offset: str,
    last_complete: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(dataset_class, tmp_path)
    cycle = pd.Timestamp("2026-10-26")
    boundary = cycle + timedelta(hours=6)
    reference = cycle + pd.Timedelta(fire_offset)
    monkeypatch.setattr(
        xr,
        "open_zarr",
        Mock(return_value=xr.Dataset(coords={"time": [cycle - timedelta(hours=3)]})),
    )
    monkeypatch.setattr(
        pd.Timestamp, "now", lambda *args, **kwargs: boundary + timedelta(hours=3)
    )
    jobs, template = dataset._operational_update_jobs(
        "job", tmp_path, job_reference_time=reference
    )
    assert pd.Timestamp(template.time.values[-1]) == boundary
    results: dict[str, list[SourceFileResult]] = {}
    saw_accumulated_boundary = False
    saw_missing_instantaneous_boundary = False
    for job in jobs:
        for group in job.source_file_var_groups(job.data_vars):
            coords = job.generate_source_file_coords(template.to_dataset(), group)
            for var in group:
                results.setdefault(var.path, [])
                for coord in coords:
                    assert isinstance(
                        coord, NoaaGfsSourceFileCoord | GefsAnalysisSourceFileCoord
                    )
                    status = SourceFileStatus.Succeeded
                    if coord.out_loc()["time"] == boundary:
                        if var.has_hour_0_values():
                            assert coord.init_time == boundary
                            assert coord.lead_time == timedelta(0)
                            status = SourceFileStatus.DownloadFailed
                            saw_missing_instantaneous_boundary = True
                        else:
                            assert coord.init_time == cycle
                            assert coord.lead_time == timedelta(hours=6)
                            saw_accumulated_boundary = True
                    results[var.path].append(
                        SourceFileResult(
                            status=status, out_loc=coord.out_loc(), url=coord.get_url()
                        )
                    )
    assert saw_accumulated_boundary
    assert saw_missing_instantaneous_boundary
    trimmed = jobs[-1].update_template_with_results(results)
    assert pd.Timestamp(trimmed.time.values[-1]) == cycle + pd.Timedelta(last_complete)
    assert boundary not in trimmed.time.values
    monkeypatch.setattr(pd.Timestamp, "now", lambda *args, **kwargs: reference)
    (currency,) = [
        check
        for check in dataset.validators()
        if isinstance(check, validation.CheckCurrentData)
    ]
    context = validation.ValidationContext(
        store=MemoryStore(),
        ds=trimmed.to_dataset(),
        append_dim="time",
        append_dim_frequency=dataset.template_config.append_dim_frequency,
    )
    assert currency.check(context).passed


@pytest.mark.parametrize(
    ("age", "wall_age", "extension"),
    [("27h59m59s", "29h", False), ("28h", "27h", True)],
)
def test_gefs_extension_age_gate_uses_the_planning_reference(
    age: str,
    wall_age: str,
    extension: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(GefsForecast35DayDataset, tmp_path)
    init = pd.Timestamp("2026-10-25")
    reference = init + pd.Timedelta(age)
    monkeypatch.setattr(
        xr, "open_zarr", Mock(return_value=xr.Dataset(coords={"init_time": [init]}))
    )
    monkeypatch.setattr(
        pd.Timestamp, "now", lambda *args, **kwargs: init + pd.Timedelta(wall_age)
    )
    jobs, _ = dataset._operational_update_jobs(
        "job", tmp_path, job_reference_time=reference
    )
    job = jobs[0]
    assert isinstance(job, GefsForecast35DayRegionJob)
    assert job.job_reference_time == reference
    var = job.data_vars[0]
    region = job.template_ds.to_dataset().sel(init_time=[init])
    coords = job.generate_source_file_coords(region, [var])
    assert (
        any(coord.lead_time > GEFS_PRE_EXTENSION_MAX for coord in coords) == extension
    )
