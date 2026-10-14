from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.rrfs.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaRrfsForecast18HourVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.dynamical_dataset import (
    NoaaRrfsForecast84HourVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.dynamical_dataset import (
    NoaaRrfsForecastSubHourlyVirtualDataset,
)
from reformatters.noaa.rrfs_ens.forecast_virtual.dynamical_dataset import (
    NoaaRrfsEnsForecastVirtualDataset,
)

type DatasetClass = type[
    NoaaRrfsForecast84HourVirtualDataset
    | NoaaRrfsForecast18HourVirtualDataset
    | NoaaRrfsForecastSubHourlyVirtualDataset
    | NoaaRrfsEnsForecastVirtualDataset
]

CLASSES = (
    NoaaRrfsForecast84HourVirtualDataset,
    NoaaRrfsForecast18HourVirtualDataset,
    NoaaRrfsForecastSubHourlyVirtualDataset,
    NoaaRrfsEnsForecastVirtualDataset,
)


@pytest.mark.parametrize("cls", CLASSES)
def test_operational_jobs_are_enabled_and_use_the_source_container(
    cls: DatasetClass, tmp_path: Path
) -> None:
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    jobs = dataset.operational_kubernetes_resources("test")
    assert len(jobs) == 1
    assert all(not job.suspend for job in jobs)
    assert all(job.dataset_id == dataset.dataset_id for job in jobs)
    assert all(job.as_kubernetes_object()["spec"]["suspend"] is False for job in jobs)
    assert dataset.icechunk_virtual_config is not None
    assert (
        dataset.icechunk_virtual_config.containers[0].url_prefix
        == "s3://noaa-rrfs-ops-pds/"
    )


@pytest.mark.parametrize(
    ("cls", "offset", "duration", "grace", "schedule"),
    [
        (
            NoaaRrfsForecast84HourVirtualDataset,
            100,
            timedelta(minutes=135),
            timedelta(minutes=5),
            "40 1,7,13,19 * * *",
        ),
        (
            NoaaRrfsForecast18HourVirtualDataset,
            100,
            timedelta(minutes=60),
            timedelta(minutes=5),
            "40 1,4,7,10,13,16,19,22 * * *",
        ),
        (
            NoaaRrfsForecastSubHourlyVirtualDataset,
            75,
            timedelta(minutes=59),
            timedelta(seconds=60),
            "15 * * * *",
        ),
        (
            NoaaRrfsEnsForecastVirtualDataset,
            75,
            timedelta(minutes=160),
            timedelta(minutes=5),
            "15 1,7,13,19 * * *",
        ),
    ],
)
def test_poll_deadline_update_schedule_and_decode_health(
    cls: DatasetClass,
    offset: int,
    duration: timedelta,
    grace: timedelta,
    schedule: str,
    tmp_path: Path,
) -> None:
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    (update,) = dataset.operational_kubernetes_resources("test")
    assert update.schedule == schedule
    assert dataset.update_offset_minutes == offset
    assert dataset.update_deadline == duration
    if cls is not NoaaRrfsForecastSubHourlyVirtualDataset:
        assert (
            "activeDeadlineSeconds"
            not in update.as_kubernetes_object()["spec"]["jobTemplate"]["spec"]
        )
    assert update.cpu == "1.5"
    assert update.memory == "3.7G"
    init = pd.Timestamp("2026-10-02T00")
    fire = init + pd.Timedelta(minutes=dataset.update_offset_minutes)
    end = fire + dataset.update_deadline
    assert update.previous_fire_time(fire) == fire
    assert update.pod_active_deadline == duration
    assert dataset._virtual_poll_deadline(fire) == end - grace
    assert dataset._virtual_poll_deadline(fire + pd.Timedelta("10min")) == end - grace
    current, completeness, decode = dataset.validators()
    assert isinstance(current, validation.CheckCurrentData)
    assert current.max_delay == (pd.Timedelta(minutes=offset) + duration - grace)
    assert isinstance(completeness, validation.CheckVirtualManifestCompleteness)
    assert completeness.min_present_fraction == (
        (6 / 18, 1.0) if cls is NoaaRrfsForecastSubHourlyVirtualDataset else (1.0,)
    )
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    assert decode.max_workers == 2
    assert decode.allow_all_nan_vars == (
        ("new_snow_density_surface",)
        if cls
        in {NoaaRrfsForecast18HourVirtualDataset, NoaaRrfsForecast84HourVirtualDataset}
        else ()
    )


def test_sub_hourly_deadline_budget(tmp_path: Path) -> None:
    dataset = NoaaRrfsForecastSubHourlyVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    (update,) = dataset.operational_kubernetes_resources("test")
    pod = update.as_kubernetes_object()["spec"]["jobTemplate"]["spec"]["template"][
        "spec"
    ]
    assert pod["activeDeadlineSeconds"] == 3540
    job_spec = update.as_kubernetes_object()["spec"]["jobTemplate"]["spec"]
    assert job_spec["activeDeadlineSeconds"] == 3540
    fire = pd.Timestamp("2026-10-11T01:15")
    next_fire = fire + pd.Timedelta("1h")
    termination_end = (
        fire
        + pd.Timedelta(seconds=job_spec["activeDeadlineSeconds"])
        + pd.Timedelta(seconds=pod["terminationGracePeriodSeconds"])
    )
    assert next_fire - termination_end == pd.Timedelta(seconds=30)
    assert dataset._virtual_poll_deadline(fire) == pd.Timestamp("2026-10-11T02:13:00")
    assert (
        fire + update.pod_active_deadline - dataset._virtual_poll_deadline(fire)
    ) == pd.Timedelta(seconds=60)
