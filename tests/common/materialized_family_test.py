import inspect
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import ReformatCronJob
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.validation import CheckCurrentData
from tests.common.current_data_deadline_test import _check_at_time

_ADOPTIONS = {
    "ecmwf-ifs-ens-forecast-15-day-0-25-degree": ("5 8 * * *", "8h40m", 24),
    "ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree": ("5 10 * * *", "58h5m", 24),
    "ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree": ("5 10 * * *", "58h5m", 24),
    "ecmwf-aifs-single-forecast": ("39 5,11,17,23 * * *", "5h39m", 6),
    "ecmwf-aifs-ens-forecast": ("59 5,11,17,23 * * *", "5h59m", 6),
    "dwd-icon-eu-forecast-5-day": ("52 3,9,15,21 * * *", "3h52m", 6),
    "eccc-hrdps-forecast": ("30 4,10,16,22 * * *", "4h30m", 6),
    "nasa-imerg-analysis-early": ("38 * * * *", "8h", 1),
    "nasa-imerg-analysis-late": ("44 * * * *", "18h", 1),
    "u-arizona-swann-analysis": ("0 20 * * *", "5D", 24),
    "nasa-smap-level3-36km-v9": ("0 6,18 * * *", "6D", 12),
    "ucsb-chc-chirps-analysis-final": ("0 0 */3 * *", "60D", 24),
    "ucsb-chc-chirps-analysis-preliminary": ("0 19 * * *", "10D", 24),
    "noaa-ndvi-cdr-analysis": ("0 20 * * *", "30D", 24),
}


_MATERIALIZED_DATASETS = [
    ds
    for ds in DYNAMICAL_DATASETS
    if issubclass(ds.region_job_class, MaterializedRegionJob)
]
_DERIVED_ENDS = {
    "noaa-gfs-analysis": "2026-10-26T19:00",
    "noaa-gefs-analysis": "2026-10-26T21:00",
    "ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree": "2026-10-25",
    "ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree": "2026-10-25",
    "ecmwf-aifs-ens-forecast": "2026-10-26T08:50",
}


def dataset_for(dataset_id: str) -> DynamicalDataset:
    return next(ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == dataset_id)


@pytest.mark.parametrize("dataset_id", _ADOPTIONS)
def test_materialized_family_budgets_and_timing(dataset_id: str) -> None:
    dataset = dataset_for(dataset_id)
    schedule, delay, hours = _ADOPTIONS[dataset_id]
    (cron,) = [
        job
        for job in dataset.operational_kubernetes_resources("test")
        if isinstance(job, ReformatCronJob)
    ]
    assert cron.schedule == schedule
    assert cron.concurrency_policy == "Allow"
    assert cron.pod_active_deadline == timedelta(hours=min(hours, 6))
    assert cron.active_deadline == timedelta(hours=min(hours, 6))
    (currency,) = [v for v in dataset.validators() if isinstance(v, CheckCurrentData)]
    assert currency.max_delay == pd.Timedelta(delay)
    rendered = cron.as_kubernetes_object()["spec"]["jobTemplate"]["spec"]
    assert rendered["activeDeadlineSeconds"] == min(hours, 6) * 3600
    assert rendered["template"]["spec"]["activeDeadlineSeconds"] == min(hours, 6) * 3600


@pytest.mark.parametrize(
    "dataset", _MATERIALIZED_DATASETS, ids=lambda ds: ds.dataset_id
)
def test_materialized_builders_require_reference_time(
    dataset: DynamicalDataset,
) -> None:
    parameter = inspect.signature(
        dataset.region_job_class.operational_update_jobs
    ).parameters["job_reference_time"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "dataset", _MATERIALIZED_DATASETS, ids=lambda ds: ds.dataset_id
)
def test_materialized_builders_use_reference_not_wall_clock(
    dataset: DynamicalDataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job_class = dataset.region_job_class
    assert issubclass(job_class, MaterializedRegionJob)
    append_dim = dataset.template_config.append_dim
    variables = dataset.template_config.data_vars
    existing = xr.Dataset(
        coords={
            append_dim: pd.date_range(
                "2026-10-20",
                periods=2,
                freq=dataset.template_config.append_dim_frequency,
            )
        }
    )
    reference = pd.Timestamp("2026-10-26T14:20")
    template = Mock(return_value=xr.DataTree(existing))
    monkeypatch.setattr(xr, "open_zarr", Mock(return_value=existing))
    monkeypatch.setattr(job_class, "get_jobs", Mock(return_value=[]))
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        Mock(side_effect=AssertionError("Builder must use its explicit reference")),
    )
    job_class.operational_update_jobs(
        primary_store=MemoryStore(),
        tmp_store=tmp_path,
        get_template_fn=template,
        append_dim=append_dim,
        all_data_vars=variables,
        reformat_job_name="test",
        job_reference_time=reference,
    )
    expected = pd.Timestamp(_DERIVED_ENDS.get(dataset.dataset_id, reference))
    template.assert_called_once_with(expected)


@pytest.mark.parametrize(
    ("dataset_id", "offset"),
    [
        ("ecmwf-aifs-single-forecast", "5h39m"),
        ("ecmwf-aifs-ens-forecast", "5h59m"),
        ("eccc-hrdps-forecast", "4h30m"),
        ("dwd-icon-eu-forecast-5-day", "3h52m"),
        ("ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree", "58h5m"),
        ("ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree", "58h5m"),
    ],
)
def test_adopted_forecast_currency_requires_init_at_its_full_fire(
    dataset_id: str, offset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = dataset_for(dataset_id)
    init = pd.Timestamp("2026-10-20")
    fire = init + pd.Timedelta(offset)
    (cron,) = [
        job
        for job in dataset.operational_kubernetes_resources("test")
        if isinstance(job, ReformatCronJob)
    ]
    assert cron.previous_fire_time(fire) == fire
    previous = init - dataset.template_config.append_dim_frequency
    assert _check_at_time(
        dataset_id, fire - timedelta(seconds=1), previous, monkeypatch
    ).passed
    assert not _check_at_time(dataset_id, fire, previous, monkeypatch).passed
    assert _check_at_time(dataset_id, fire, init, monkeypatch).passed


def test_materialized_allow_deadlines_cap_minimum_calendar_fire_interval() -> None:
    calendar_start = pd.Timestamp("2023-01-01")
    calendar_end = pd.Timestamp("2025-01-01")
    for dataset in _MATERIALIZED_DATASETS:
        for cron in dataset.operational_kubernetes_resources("test"):
            if (
                not isinstance(cron, ReformatCronJob)
                or cron.concurrency_policy != "Allow"
            ):
                continue
            fire = cron.next_fire_time(calendar_start - timedelta(seconds=1))
            intervals = []
            while fire < calendar_end:
                next_fire = cron.next_fire_time(fire)
                intervals.append(next_fire - fire)
                fire = next_fire
            assert (
                cron.pod_active_deadline
                == cron.active_deadline
                == min(timedelta(hours=6), *intervals)
            ), dataset.dataset_id


def test_chirps_calendar_deadlines_are_capped_below_shortest_interval() -> None:
    dataset = dataset_for("ucsb-chc-chirps-analysis-final")
    (cron,) = dataset.operational_kubernetes_resources("test")
    boundary = pd.Timestamp("2026-11-01")
    assert cron.previous_fire_time(boundary) - cron.previous_fire_time(
        boundary - timedelta(seconds=1)
    ) == timedelta(days=1)
    assert cron.previous_fire_time(
        boundary + timedelta(days=3)
    ) - boundary == timedelta(days=3)
    assert cron.pod_active_deadline == cron.active_deadline == timedelta(hours=6)


def test_deprecated_plain_zarr_updater_keeps_legacy_lifecycle() -> None:
    dataset = dataset_for("ecmwf-ifs-ens-15-day-deprecated-zarr-v3")
    (cron,) = dataset.operational_kubernetes_resources("test")
    assert cron.schedule == "45 8 * * *"
    assert cron.concurrency_policy == "Replace"
    assert cron.pod_active_deadline == timedelta(minutes=45)
    assert cron.active_deadline is None
