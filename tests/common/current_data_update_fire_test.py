from datetime import timedelta

import pandas as pd
import pytest
import xarray as xr
import zarr.storage

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import validation
from reformatters.common.kubernetes import ReformatCronJob

_DUE_AT_UPDATE = [
    ("noaa-gfs-forecast", "2026-09-27 05:38", "2026-09-27 00:00", "5h38m"),
    ("noaa-gfs-analysis", "2026-09-27 03:40", "2026-09-26 20:00", "7h"),
    ("noaa-gfs-analysis-virtual", "2026-09-27 03:29", "2026-09-26 22:00", "4h45m"),
    ("noaa-gfs-forecast-virtual", "2026-09-27 03:29", "2026-09-26 18:00", "6h30m"),
    ("noaa-gefs-analysis", "2026-09-27 03:51", "2026-09-26 15:00", "12h"),
    ("noaa-gefs-forecast-35-day", "2026-09-27 06:45", "2026-09-27 00:00", "6h45m"),
    (
        "noaa-gefs-analysis-0-25-degree-virtual",
        "2026-09-27 03:45",
        "2026-09-27 00:00",
        "3h40m",
    ),
    (
        "noaa-gefs-forecast-10-day-0-25-degree-virtual",
        "2026-09-27 03:45",
        "2026-09-27 00:00",
        "3h40m",
    ),
    (
        "noaa-gefs-forecast-16-day-0-5-degree-virtual",
        "2026-09-27 03:45",
        "2026-09-27 00:00",
        "3h40m",
    ),
    (
        "noaa-gefs-forecast-35-day-0-5-degree-virtual",
        "2026-09-27 03:45",
        "2026-09-27 00:00",
        "3h40m",
    ),
    ("noaa-hrrr-forecast-48-hour", "2026-09-27 01:53", "2026-09-27 00:00", "1h53m"),
    ("noaa-hrrr-analysis", "2026-09-27 00:57", "2026-09-26 21:00", "3h40m"),
    ("noaa-hrrr-analysis-virtual", "2026-09-27 00:50", "2026-09-26 23:00", "1h30m"),
    (
        "noaa-hrrr-forecast-48-hour-virtual",
        "2026-09-27 00:50",
        "2026-09-27 00:00",
        "50m",
    ),
    (
        "noaa-hrrr-forecast-18-hour-virtual",
        "2026-09-27 00:50",
        "2026-09-27 00:00",
        "50m",
    ),
    ("noaa-mrms-conus-analysis-hourly", "2026-09-27 00:03", "2026-09-27 00:00", "3m"),
    (
        "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "2026-09-27 08:05",
        "2026-09-27 00:00",
        "8h05m",
    ),
    (
        "ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree",
        "2026-09-27 09:00",
        "2026-09-23 00:00",
        "4D",
    ),
    (
        "ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree",
        "2026-09-27 10:00",
        "2026-09-23 00:00",
        "4D",
    ),
    ("ecmwf-aifs-single-forecast", "2026-09-27 00:13", "2026-09-26 18:00", "6h13m"),
    (
        "ecmwf-aifs-single-forecast-virtual",
        "2026-09-27 05:20",
        "2026-09-27 00:00",
        "5h20m",
    ),
    ("ecmwf-aifs-ens-forecast", "2026-09-27 01:00", "2026-09-26 18:00", "7h"),
    ("dwd-icon-eu-forecast-5-day", "2026-09-27 03:52", "2026-09-27 00:00", "3h52m"),
    ("eccc-hrdps-forecast", "2026-09-27 04:30", "2026-09-27 00:00", "4h30m"),
    (
        "google-weathernext2-forecast-operational-virtual",
        "2026-09-27 01:05",
        "2026-09-26 12:00",
        "12h",
    ),
    ("nasa-imerg-analysis-early", "2026-09-27 00:38", "2026-09-26 16:30", "8h"),
    ("nasa-imerg-analysis-late", "2026-09-27 00:44", "2026-09-26 06:30", "18h"),
    (
        "ucsb-chc-chirps-analysis-final",
        "2026-09-28 00:00",
        "2026-07-30 00:00",
        "60D",
    ),
    (
        "ucsb-chc-chirps-analysis-preliminary",
        "2026-09-27 19:00",
        "2026-09-17 00:00",
        "10D",
    ),
    ("u-arizona-swann-analysis", "2026-09-27 20:00", "2026-09-22 00:00", "5D"),
    ("noaa-ndvi-cdr-analysis", "2026-09-27 20:00", "2026-08-28 00:00", "30D"),
    ("nasa-smap-level3-36km-v9", "2026-09-27 06:00", "2026-09-21 00:00", "6D"),
]


def _assert_scheduled_fire(dataset_id: str, fire: pd.Timestamp) -> None:
    dataset = next(d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id)
    update = next(
        job
        for job in dataset.operational_kubernetes_resources("test-image-tag")
        if isinstance(job, ReformatCronJob)
    )
    assert update.previous_fire_time(fire) == fire


def _check_at_fire(
    dataset_id: str,
    fire: pd.Timestamp,
    latest: pd.Timestamp,
    monkeypatch: pytest.MonkeyPatch,
) -> validation.ValidationResult:
    dataset = next(d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id)
    (current_data,) = [
        check
        for check in dataset.validators()
        if isinstance(check, validation.CheckCurrentData)
    ]
    frequency = pd.Timedelta(dataset.template_config.append_dim_frequency)
    start = pd.Timestamp(dataset.template_config.append_dim_start)
    assert latest >= start
    assert (latest - start) % frequency == timedelta(0)
    positions = pd.date_range(latest - 4 * frequency, latest, freq=frequency)
    context = validation.ValidationContext(
        store=zarr.storage.MemoryStore(),
        ds=xr.Dataset(coords={dataset.template_config.append_dim: positions}),
        append_dim=dataset.template_config.append_dim,
        append_dim_frequency=frequency,
    )
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *args, **kwargs: fire))
    return current_data.check(context)


@pytest.mark.parametrize(
    ("dataset_id", "fire", "due", "max_delay"),
    _DUE_AT_UPDATE,
    ids=[case[0] for case in _DUE_AT_UPDATE],
)
def test_current_data_due_at_update_fire(
    dataset_id: str,
    fire: str,
    due: str,
    max_delay: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = next(d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id)
    (current_data,) = [
        check
        for check in dataset.validators()
        if isinstance(check, validation.CheckCurrentData)
    ]
    assert current_data.max_delay == pd.Timedelta(max_delay)

    frequency = pd.Timedelta(dataset.template_config.append_dim_frequency)
    due_position = pd.Timestamp(due)
    update_fire = pd.Timestamp(fire)
    _assert_scheduled_fire(dataset_id, update_fire)
    assert _check_at_fire(dataset_id, update_fire, due_position, monkeypatch).passed
    stale = _check_at_fire(
        dataset_id, update_fire, due_position - frequency, monkeypatch
    )
    assert not stale.passed
    assert due_position.isoformat() in stale.message


def test_current_data_cases_cover_registered_datasets() -> None:
    registered = {
        dataset.dataset_id
        for dataset in DYNAMICAL_DATASETS
        if any(
            isinstance(check, validation.CheckCurrentData)
            for check in dataset.validators()
        )
    }
    assert {case[0] for case in _DUE_AT_UPDATE} == registered


@pytest.mark.parametrize(
    ("hour", "due_hour"),
    [(0, 21), (3, 0), (6, 3), (9, 6), (12, 9), (15, 12), (18, 15), (21, 18)],
)
def test_hrrr_analysis_every_update_fire_alignment(
    hour: int, due_hour: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    fire = pd.Timestamp("2026-09-27") + timedelta(hours=hour, minutes=57)
    due = fire.normalize() + timedelta(hours=due_hour)
    if hour == 0:
        due -= timedelta(days=1)
    _assert_scheduled_fire("noaa-hrrr-analysis", fire)
    assert _check_at_fire("noaa-hrrr-analysis", fire, due, monkeypatch).passed
    assert not _check_at_fire(
        "noaa-hrrr-analysis", fire, due - timedelta(hours=1), monkeypatch
    ).passed


@pytest.mark.parametrize(
    ("fire", "due"),
    [
        ("2026-10-31 00:00", "2026-09-01 00:00"),
        ("2026-11-01 00:00", "2026-09-02 00:00"),
    ],
)
def test_chirps_final_month_boundary(
    fire: str, due: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_scheduled_fire("ucsb-chc-chirps-analysis-final", pd.Timestamp(fire))
    assert _check_at_fire(
        "ucsb-chc-chirps-analysis-final",
        pd.Timestamp(fire),
        pd.Timestamp(due),
        monkeypatch,
    ).passed


def test_mrms_current_data_after_early_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fire = pd.Timestamp("2026-09-27 05:04")
    due = pd.Timestamp("2026-09-27 05:00")
    _assert_scheduled_fire(
        "noaa-mrms-conus-analysis-hourly", fire - timedelta(minutes=1)
    )
    assert _check_at_fire(
        "noaa-mrms-conus-analysis-hourly", fire, due, monkeypatch
    ).passed
    stale = _check_at_fire(
        "noaa-mrms-conus-analysis-hourly", fire, due - timedelta(hours=1), monkeypatch
    )
    assert not stale.passed
    assert due.isoformat() in stale.message
