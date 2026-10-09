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
def test_operational_jobs_are_suspended_and_use_the_source_container(
    cls: DatasetClass, tmp_path: Path
) -> None:
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    jobs = dataset.operational_kubernetes_resources("test")
    assert len(jobs) == 1
    assert all(job.suspend for job in jobs)
    assert all(job.dataset_id == dataset.dataset_id for job in jobs)
    assert all(job.as_kubernetes_object()["spec"]["suspend"] is True for job in jobs)
    assert dataset.icechunk_virtual_config is not None
    assert (
        dataset.icechunk_virtual_config.containers[0].url_prefix
        == "s3://noaa-rrfs-ops-pds/"
    )


@pytest.mark.parametrize(
    ("cls", "offset", "duration", "schedule"),
    [
        (NoaaRrfsForecast84HourVirtualDataset, 100, 135, "40 1,7,13,19 * * *"),
        (
            NoaaRrfsForecast18HourVirtualDataset,
            100,
            60,
            "40 1,4,7,10,13,16,19,22 * * *",
        ),
        (
            NoaaRrfsForecastSubHourlyVirtualDataset,
            75,
            55,
            "15 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 * * *",
        ),
        (NoaaRrfsEnsForecastVirtualDataset, 75, 160, "15 1,7,13,19 * * *"),
    ],
)
def test_poll_deadline_update_schedule_and_decode_health(
    cls: DatasetClass, offset: int, duration: int, schedule: str, tmp_path: Path
) -> None:
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    (update,) = dataset.operational_kubernetes_resources("test")
    assert update.schedule == schedule
    assert dataset.update_offset_minutes == offset
    assert dataset.update_deadline_minutes == duration
    assert update.cpu == "1.5"
    assert update.memory == "3.7G"
    init = pd.Timestamp("2026-10-02T00")
    fire = init + pd.Timedelta(minutes=dataset.update_offset_minutes)
    end = fire + pd.Timedelta(minutes=dataset.update_deadline_minutes)
    assert update.previous_fire_time(fire) == fire
    assert update.pod_active_deadline == pd.Timedelta(
        minutes=dataset.update_deadline_minutes
    )
    assert dataset._virtual_poll_deadline(fire) == end - pd.Timedelta("5min")
    assert dataset._virtual_poll_deadline(
        fire + pd.Timedelta("10min")
    ) == end - pd.Timedelta("5min")
    current, completeness, decode = dataset.validators()
    assert isinstance(current, validation.CheckCurrentData)
    assert current.max_delay == (
        pd.Timedelta(minutes=offset + duration) - dataset.virtual_poll_deadline_grace
    )
    assert isinstance(completeness, validation.CheckVirtualManifestCompleteness)
    assert completeness.min_present_fraction == (1.0,)
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    assert decode.max_workers == 2
    assert decode.allow_all_nan_vars == ()
