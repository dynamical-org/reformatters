from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr
from kubernetes import client
from zarr.storage import MemoryStore

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import validation
from reformatters.common.kubernetes import create_job_from_cronjob
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.types import DatetimeLike
from reformatters.noaa.refs.forecast_3_hourly_virtual.dynamical_dataset import (
    NoaaRefsForecast3HourlyVirtualDataset,
)
from reformatters.noaa.refs.forecast_hourly_virtual.dynamical_dataset import (
    NoaaRefsForecastHourlyVirtualDataset,
)
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
    | NoaaRefsForecastHourlyVirtualDataset
    | NoaaRefsForecast3HourlyVirtualDataset
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
    ("cls", "offset", "duration", "grace", "schedule", "cutoff_minutes"),
    [
        (
            NoaaRrfsForecast84HourVirtualDataset,
            65,
            timedelta(minutes=170),
            timedelta(minutes=5),
            "5 1,7,13,19 * * *",
            230,
        ),
        (
            NoaaRrfsForecast18HourVirtualDataset,
            65,
            timedelta(minutes=95),
            timedelta(minutes=5),
            "5 1,4,7,10,13,16,19,22 * * *",
            155,
        ),
        (
            NoaaRrfsForecastSubHourlyVirtualDataset,
            75,
            timedelta(minutes=59),
            timedelta(seconds=60),
            "15 * * * *",
            133,
        ),
        (
            NoaaRrfsEnsForecastVirtualDataset,
            70,
            timedelta(minutes=192),
            timedelta(minutes=6),
            "10 1,7,13,19 * * *",
            256,
        ),
    ],
)
def test_poll_deadline_update_schedule_and_decode_health(
    cls: DatasetClass,
    offset: int,
    duration: timedelta,
    grace: timedelta,
    schedule: str,
    cutoff_minutes: int,
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
    spec = update.as_kubernetes_object()["spec"]
    assert spec["concurrencyPolicy"] == "Replace"
    job = spec["jobTemplate"]["spec"]
    pod = job["template"]["spec"]
    assert pod["activeDeadlineSeconds"] == int(duration.total_seconds())
    assert job["activeDeadlineSeconds"] == (
        359 * 60 if dataset.retry_late_sources else int(duration.total_seconds())
    )
    assert update.cpu == "1.5"
    assert update.memory == "3.7G"
    init = pd.Timestamp("2026-10-02T00")
    fire = init + pd.Timedelta(minutes=dataset.update_offset_minutes)
    end = fire + dataset.update_deadline
    assert update.previous_fire_time(fire) == fire
    assert update.pod_active_deadline == duration
    cutoff = init + pd.Timedelta(minutes=cutoff_minutes)
    assert dataset.virtual_poll_deadline_grace == grace
    assert dataset._virtual_poll_deadline(fire) == end - grace == cutoff
    termination_end = end + pd.Timedelta(seconds=pod["terminationGracePeriodSeconds"])
    assert (
        update.next_fire_time(fire)
        == fire + dataset.template_config.append_dim_frequency
    )
    assert (
        update.next_fire_time(fire) - termination_end
        == (
            dataset.template_config.append_dim_frequency
            - duration
            - pd.Timedelta(seconds=30)
        )
        > pd.Timedelta(0)
    )
    current, completeness, decode = dataset.validators()
    assert isinstance(current, validation.CheckCurrentData)
    assert current.max_delay == pd.Timedelta(minutes=cutoff_minutes)
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


@pytest.mark.parametrize(
    "cls",
    [
        *CLASSES,
        NoaaRefsForecastHourlyVirtualDataset,
        NoaaRefsForecast3HourlyVirtualDataset,
    ],
)
@pytest.mark.parametrize("across_midnight", [False, True])
def test_consecutive_fires_keep_init_selection_and_restart_cutoff(
    cls: DatasetClass,
    across_midnight: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    (update,) = dataset.operational_kubernetes_resources("test")
    cadence = dataset.template_config.append_dim_frequency
    init = pd.Timestamp("2026-10-11")
    if across_midnight:
        init += pd.Timedelta("24h") - cadence

    def get_template(self: object, end: DatetimeLike) -> xr.DataTree:
        return xr.DataTree(
            xr.Dataset(
                coords={
                    "init_time": pd.date_range(
                        init - pd.Timedelta("24h"), end, freq=cadence, inclusive="left"
                    )
                }
            )
        )

    monkeypatch.setattr(cls, "_get_template", get_template)
    monkeypatch.setattr(
        type(dataset.store_factory), "primary_store", lambda self: MemoryStore()
    )
    for due_init in pd.date_range(init, periods=3, freq=cadence):
        fire = due_init + pd.Timedelta(minutes=dataset.update_offset_minutes)
        assert update.previous_fire_time(fire) == fire
        cutoff = fire + dataset.update_deadline - dataset.virtual_poll_deadline_grace
        for now in (fire, fire + pd.Timedelta("10min"), cutoff + pd.Timedelta("1s")):
            monkeypatch.setattr(
                pd.Timestamp, "now", classmethod(lambda cls, now=now: now)
            )
            assert update.previous_fire_time(now) == fire
            assert dataset._virtual_poll_deadline(now) == cutoff
            (job,), _ = dataset._operational_update_jobs("test", tmp_path)
            expected = pd.date_range(
                end=due_init, periods=int(pd.Timedelta("18h") / cadence), freq=cadence
            )
            actual = (
                job.template_ds.to_dataset()
                .isel(init_time=job.region)
                .get_index("init_time")
            )
            pd.testing.assert_index_equal(actual, expected.rename("init_time"))


@pytest.mark.parametrize(
    ("variant", "retry_minute", "created", "poll_minutes_remaining"),
    [
        ("rrfs-forecast-18-hour", 146, True, 9),
        ("rrfs-forecast-18-hour", 150, False, 5),
        ("rrfs-forecast-18-hour", 158, False, -3),
        ("rrfs-forecast-84-hour", 233, True, -3),
        ("rrfs-forecast-sub-hourly", 134, False, -1),
    ],
)
def test_rrfs_retry_admission_and_fire_anchored_cutoff(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    retry_minute: int,
    created: bool,
    poll_minutes_remaining: int,
) -> None:
    dataset = next(
        dataset
        for dataset in DYNAMICAL_DATASETS
        if dataset.dataset_id == f"noaa-{variant}-virtual"
    )
    (update,) = dataset.operational_kubernetes_resources("test")
    spec = update.as_kubernetes_object()["spec"]["jobTemplate"]["spec"]
    cronjob = Mock()
    cronjob.metadata.uid = "rrfs-cronjob-uid"
    cronjob.spec.suspend = False
    cronjob.spec.schedule = update.schedule
    cronjob.spec.job_template.metadata = None
    cronjob.spec.job_template.spec = client.V1JobSpec(
        active_deadline_seconds=spec["activeDeadlineSeconds"],
        template=client.V1PodTemplateSpec(
            spec=client.V1PodSpec(
                containers=[client.V1Container(name="worker")],
                active_deadline_seconds=spec["template"]["spec"][
                    "activeDeadlineSeconds"
                ],
            )
        ),
    )
    now = pd.Timestamp("2026-10-15T00:00:00Z") + timedelta(minutes=retry_minute)
    batch = Mock()
    batch.read_namespaced_cron_job.return_value = cronjob
    monkeypatch.setenv("POD_NAMESPACE", "weather")
    monkeypatch.setattr(
        "reformatters.common.kubernetes.config.load_incluster_config", Mock()
    )
    monkeypatch.setattr(
        "reformatters.common.kubernetes.client.BatchV1Api", lambda: batch
    )
    monkeypatch.setattr(
        "reformatters.common.kubernetes.datetime", Mock(now=lambda tz: now)
    )

    assert (
        create_job_from_cronjob(
            update.name, "weather-retry-r1", skip_if_next_run_within_deadline=True
        )
        is created
    )
    assert batch.create_namespaced_job.called is created
    assert dataset._virtual_poll_deadline(now) - now == timedelta(
        minutes=poll_minutes_remaining
    )
    if created:
        retry = batch.create_namespaced_job.call_args.args[1]
        assert retry.spec.active_deadline_seconds == spec["activeDeadlineSeconds"]
        assert retry.spec.template.spec.active_deadline_seconds == int(
            update.pod_active_deadline.total_seconds()
        )
