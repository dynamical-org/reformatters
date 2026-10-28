from datetime import timedelta

import pandas as pd
import pytest
import xarray as xr
import zarr.storage

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import validation
from reformatters.common.kubernetes import ReformatCronJob
from reformatters.common.virtual_region_job import VirtualRegionJob
from reformatters.noaa.rrfs.dynamical_dataset import NoaaRrfsDataset

_FRESHNESS_DEADLINES = [
    ("noaa-gfs-forecast", "2026-09-27 00:00", "5h44m"),
    ("noaa-gfs-analysis", "2026-09-26 20:00", "3h58m"),
    ("noaa-gfs-analysis-virtual", "2026-09-26 22:00", "3h57m"),
    ("noaa-gfs-forecast-virtual", "2026-09-26 18:00", "3h57m"),
    ("noaa-gefs-analysis", "2026-09-26 15:00", "4h03m"),
    ("noaa-gefs-forecast-35-day", "2026-09-27 00:00", "7h15m"),
    ("noaa-gefs-analysis-0-25-degree-virtual", "2026-09-27 00:00", "4h09m"),
    ("noaa-gefs-forecast-10-day-0-25-degree-virtual", "2026-09-27 00:00", "4h09m"),
    ("noaa-gefs-forecast-16-day-0-5-degree-virtual", "2026-09-27 00:00", "4h08m"),
    ("noaa-gefs-forecast-35-day-0-5-degree-virtual", "2026-09-27 00:00", "3h50m"),
    ("noaa-hrrr-forecast-48-hour", "2026-09-27 00:00", "2h12m"),
    ("noaa-hrrr-analysis", "2026-09-26 21:00", "3h10m"),
    ("noaa-hrrr-analysis-virtual", "2026-09-26 23:00", "80m"),
    ("noaa-rtma-analysis-virtual", "2026-09-27 00:00", "73m"),
    ("noaa-rtma-ru-analysis-virtual", "2026-09-27 00:15", "45m"),
    ("noaa-urma-analysis-virtual", "2026-09-27 00:00", "7h40m"),
    ("noaa-hrrr-forecast-48-hour-virtual", "2026-09-27 00:00", "74m"),
    ("noaa-hrrr-forecast-18-hour-virtual", "2026-09-27 00:00", "74m"),
    ("noaa-rrfs-forecast-84-hour-virtual", "2026-09-27 00:00", "133m"),
    ("noaa-rrfs-forecast-18-hour-virtual", "2026-09-27 00:00", "133m"),
    ("noaa-rrfs-forecast-sub-hourly-virtual", "2026-09-27 00:00", "138m"),
    ("noaa-rrfs-ens-forecast-virtual", "2026-09-27 00:00", "122m"),
    ("noaa-refs-forecast-hourly-virtual", "2026-09-27 00:00", "150m"),
    ("noaa-refs-forecast-3-hourly-virtual", "2026-09-27 00:00", "2h36m"),
    ("noaa-mrms-conus-analysis-hourly", "2026-09-27 00:00", "5m"),
    ("ecmwf-ifs-ens-forecast-15-day-0-25-degree", "2026-09-27 00:00", "8h40m"),
    ("ecmwf-ifs-ens-15-day-deprecated-zarr-v3", "2026-09-27 00:00", "9h20m"),
    ("ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree", "2026-09-23 00:00", "58h05m"),
    ("ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree", "2026-09-23 00:00", "58h05m"),
    ("ecmwf-aifs-single-forecast", "2026-09-26 18:00", "5h39m"),
    ("ecmwf-aifs-single-forecast-virtual", "2026-09-27 00:00", "6h06m"),
    ("ecmwf-aifs-ens-forecast", "2026-09-26 18:00", "5h59m"),
    ("dwd-icon-eu-forecast-5-day", "2026-09-27 00:00", "3h52m"),
    ("eccc-hrdps-forecast", "2026-09-27 00:00", "4h30m"),
    ("google-weathernext2-forecast-operational-virtual", "2026-09-26 12:00", "12h"),
    (
        "google-weathernext3-forecast-15-day-0-1-degree-virtual",
        "2026-09-26 12:00",
        "12h",
    ),
    (
        "google-weathernext3-forecast-48-hour-0-1-degree-virtual",
        "2026-09-26 13:00",
        "12h",
    ),
    (
        "google-weathernext3-forecast-15-day-0-05-degree-virtual",
        "2026-09-26 12:00",
        "12h",
    ),
    (
        "google-weathernext3-forecast-48-hour-0-05-degree-virtual",
        "2026-09-26 13:00",
        "12h",
    ),
    ("nasa-imerg-analysis-early", "2026-09-26 16:30", "8h"),
    ("nasa-imerg-analysis-late", "2026-09-26 06:30", "18h"),
    ("ucsb-chc-chirps-analysis-final", "2026-07-30 00:00", "60D"),
    ("ucsb-chc-chirps-analysis-preliminary", "2026-09-17 00:00", "10D"),
    ("u-arizona-swann-analysis", "2026-09-22 00:00", "5D"),
    ("noaa-ndvi-cdr-analysis", "2026-08-28 00:00", "30D"),
    ("nasa-smap-level3-36km-v9", "2026-09-21 00:00", "6D"),
]


def _check_at_time(
    dataset_id: str,
    now: pd.Timestamp,
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
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now))
    return current_data.check(context)


@pytest.mark.parametrize(
    ("dataset_id", "position", "max_delay"),
    _FRESHNESS_DEADLINES,
    ids=[case[0] for case in _FRESHNESS_DEADLINES],
)
def test_current_data_deadline(
    dataset_id: str,
    position: str,
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
    due = pd.Timestamp(position)
    deadline = due + pd.Timedelta(max_delay)
    previous = due - frequency
    assert _check_at_time(
        dataset_id, deadline - timedelta(seconds=1), previous, monkeypatch
    ).passed
    for now in (deadline, deadline + timedelta(seconds=1)):
        stale = _check_at_time(dataset_id, now, previous, monkeypatch)
        assert not stale.passed
        assert due.isoformat() in stale.message
        assert _check_at_time(dataset_id, now, due, monkeypatch).passed


def test_current_data_cases_cover_registered_datasets() -> None:
    registered = {
        dataset.dataset_id
        for dataset in DYNAMICAL_DATASETS
        if any(
            isinstance(check, validation.CheckCurrentData)
            for check in dataset.validators()
        )
    }
    assert {case[0] for case in _FRESHNESS_DEADLINES} == registered


@pytest.mark.parametrize("hour", [0, 6, 12, 18])
@pytest.mark.parametrize(
    ("dataset_id", "validation_delay"),
    [
        ("ecmwf-aifs-single-forecast", "8h15m"),
        ("ecmwf-aifs-single-forecast-virtual", "6h06m"),
        ("ecmwf-aifs-single-forecast-virtual", "7h35m"),
        ("ecmwf-aifs-single-forecast-virtual", "8h38m"),
        ("ecmwf-aifs-ens-forecast", "8h"),
        ("ecmwf-aifs-ens-forecast", "8h08m"),
    ],
)
def test_aifs_validation_requires_the_current_init(
    dataset_id: str,
    validation_delay: str,
    hour: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init = pd.Timestamp("2026-10-21") + timedelta(hours=hour)
    now = init + pd.Timedelta(validation_delay)
    assert not _check_at_time(
        dataset_id, now, init - timedelta(hours=6), monkeypatch
    ).passed
    assert _check_at_time(dataset_id, now, init, monkeypatch).passed


@pytest.mark.parametrize(
    ("dataset_id", "now", "latest"),
    [
        ("noaa-hrrr-forecast-18-hour-virtual", "2026-09-27 00:51", "2026-09-26 23:00"),
        ("noaa-gfs-forecast-virtual", "2026-09-27 03:49", "2026-09-26 18:00"),
        ("noaa-mrms-conus-analysis-hourly", "2026-09-27 05:04", "2026-09-27 04:00"),
        ("noaa-hrrr-analysis", "2026-09-27 03:58", "2026-09-27 00:00"),
    ],
)
def test_current_data_allows_source_and_processing_time(
    dataset_id: str, now: str, latest: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _check_at_time(
        dataset_id, pd.Timestamp(now), pd.Timestamp(latest), monkeypatch
    ).passed


def test_virtual_polling_starts_by_freshness_deadline() -> None:
    init = pd.Timestamp("2026-09-27")
    for dataset in DYNAMICAL_DATASETS:
        if not issubclass(dataset.region_job_class, VirtualRegionJob):
            continue
        for check in dataset.validators():
            if not isinstance(check, validation.CheckCurrentData):
                continue
            update = next(
                job
                for job in dataset.operational_kubernetes_resources("test-image")
                if isinstance(job, ReformatCronJob)
            )
            assert update.next_fire_time(init) <= init + check.max_delay, (
                dataset.dataset_id
            )


@pytest.mark.parametrize(
    "dataset_id",
    [
        case[0]
        for case in _FRESHNESS_DEADLINES
        if case[0].startswith(("noaa-rrfs-", "noaa-refs-"))
        and case[0] != "noaa-rrfs-forecast-sub-hourly-virtual"
    ],
)
def test_rrfs_and_refs_current_init_is_due_at_poll_deadline(
    dataset_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = next(d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id)
    due = pd.Timestamp("2026-09-27 00:00")
    (update,) = dataset.operational_kubernetes_resources("test-image")
    assert isinstance(dataset, NoaaRrfsDataset)
    fire = due + timedelta(minutes=dataset.update_offset_minutes)
    assert update.previous_fire_time(fire) == fire
    deadline = dataset._virtual_poll_deadline(fire)
    previous = due - dataset.template_config.append_dim_frequency
    stale = _check_at_time(dataset_id, deadline, previous, monkeypatch)
    assert not stale.passed
    assert due.isoformat() in stale.message
    assert _check_at_time(dataset_id, deadline, due, monkeypatch).passed


def test_rrfs_sub_hourly_init_is_due_at_next_fire_poll_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "noaa-rrfs-forecast-sub-hourly-virtual"
    dataset = next(d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id)
    assert isinstance(dataset, NoaaRrfsDataset)
    due = pd.Timestamp("2026-09-27 00:00")
    cadence = dataset.template_config.append_dim_frequency
    deadline = dataset._virtual_poll_deadline(
        due + timedelta(minutes=dataset.update_offset_minutes)
    )
    previous = due - cadence
    assert _check_at_time(dataset_id, deadline, previous, monkeypatch).passed
    stale = _check_at_time(dataset_id, deadline + cadence, previous, monkeypatch)
    assert not stale.passed
    assert due.isoformat() in stale.message


@pytest.mark.parametrize("hour", [0, 6, 12, 18])
def test_hrrr48_backstop_requires_missing_init_at_fire_plus_startup(
    hour: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_id = "noaa-hrrr-forecast-48-hour"
    dataset = next(d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id)
    init = pd.Timestamp("2026-10-27") + timedelta(hours=hour)
    (cron,) = dataset.operational_kubernetes_resources("test")
    fire = cron.next_fire_time(init)
    assert fire == init + timedelta(hours=2, minutes=12)
    now = fire + timedelta(seconds=35)
    assert not _check_at_time(
        dataset_id, now, init - timedelta(hours=6), monkeypatch
    ).passed
    assert _check_at_time(dataset_id, now, init, monkeypatch).passed


@pytest.mark.parametrize(
    ("dataset_id", "offset", "startup", "hours"),
    [
        ("noaa-gfs-forecast", "5h44m", "21s", (0, 6, 12, 18)),
        ("noaa-gefs-forecast-35-day", "7h15m", "58s", (0,)),
        ("ecmwf-aifs-single-forecast", "7h38m", "84s", (0, 6, 12, 18)),
        ("ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree", "58h05m", "51s", (0,)),
        ("ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree", "58h05m", "46s", (0,)),
    ],
)
def test_trigger_family_backstop_requires_the_due_init(
    dataset_id: str,
    offset: str,
    startup: str,
    hours: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = next(d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id)
    (cron,) = dataset.operational_kubernetes_resources("test")
    check = next(
        check
        for check in dataset.validators()
        if isinstance(check, validation.CheckCurrentData)
    )
    assert check.max_delay == pd.Timedelta(offset)
    for hour in hours:
        init = pd.Timestamp("2026-10-20") + timedelta(hours=hour)
        fire = init + pd.Timedelta(offset)
        assert cron.previous_fire_time(fire) == fire
        previous = init - dataset.template_config.append_dim_frequency
        assert _check_at_time(
            dataset_id, fire - timedelta(seconds=1), previous, monkeypatch
        ).passed
        now = fire + pd.Timedelta(startup)
        assert not _check_at_time(dataset_id, now, previous, monkeypatch).passed
        assert _check_at_time(dataset_id, now, init, monkeypatch).passed
