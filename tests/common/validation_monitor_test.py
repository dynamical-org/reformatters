from datetime import timedelta
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import sentry_sdk
import sentry_sdk.crons

from reformatters.__main__ import DYNAMICAL_DATASETS, OPERATIONAL_ARCHIVERS
from reformatters.common import operational
from reformatters.common.config import Config
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.monitoring import monitor_cron

_DAY = pd.Timestamp("2026-10-30")
_FIRST_REPASS_BACKOFF = timedelta(minutes=5)
_SLACK = timedelta(minutes=5)


def _dataset(dataset_id: str) -> DynamicalDataset[Any, Any]:
    return next(d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id)


def _update_cron(dataset: DynamicalDataset[Any, Any]) -> CronJob:
    return next(
        cron
        for cron in dataset.operational_kubernetes_resources("image")
        if isinstance(cron, ReformatCronJob)
    )


def _validation_monitor_config(
    dataset: DynamicalDataset[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    monkeypatch.setattr(type(Config), "is_sentry_enabled", True)
    monkeypatch.delenv("CRON_JOB_NAME", raising=False)
    capture = Mock()
    monkeypatch.setattr(sentry_sdk.crons, "capture_checkin", capture)
    monkeypatch.setattr(sentry_sdk, "flush", Mock())
    operational.register_run_monitor(monitor_cron)

    with dataset._validation_monitor("job-name"):
        pass

    configs = [call.kwargs["monitor_config"] for call in capture.call_args_list]
    assert len(configs) == 2
    assert configs[0] == configs[1]
    return configs[0]


@pytest.mark.parametrize(
    ("dataset_id", "schedule", "checkin_margin"),
    [
        ("noaa-gefs-forecast-35-day", "30 5 * * *", 131),
        ("ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree", "0 3 * * *", 443),
        ("ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree", "0 3 * * *", 443),
        ("noaa-gfs-forecast", "44 5,11,17,23 * * *", 370),
        ("ecmwf-ifs-ens-forecast-15-day-0-25-degree", "5 8 * * *", 370),
    ],
)
def test_validation_monitor_config(
    dataset_id: str,
    schedule: str,
    checkin_margin: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _validation_monitor_config(_dataset(dataset_id), monkeypatch)

    assert config["schedule"] == {"type": "crontab", "value": schedule}
    assert config["checkin_margin"] == checkin_margin
    assert config["max_runtime"] == 360


# Earliest observed GEFS 00Z tier-1 (a/b <= f384) source completion; a trigger
# cannot validate before its source is ready.
_GEFS_35_DAY_SOURCE_READY_LOWER_BOUND = _DAY + pd.Timedelta("6h29m")
_IFS_46_DAY_ARCHIVER_FIRE = next(
    cron.next_fire_time(_DAY)
    for archiver in OPERATIONAL_ARCHIVERS
    for cron in archiver.operational_kubernetes_resources("image")
    if cron.name == "ecmwf-ifs-ens-46-day-gribs-archive-grib-files"
)


@pytest.mark.parametrize(
    ("dataset_id", "trigger_lower_bound", "typical_fire_to_validation"),
    [
        (
            "noaa-gefs-forecast-35-day",
            _GEFS_35_DAY_SOURCE_READY_LOWER_BOUND,
            timedelta(minutes=8),
        ),
        (
            "ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree",
            _IFS_46_DAY_ARCHIVER_FIRE,
            timedelta(minutes=4),
        ),
        (
            "ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree",
            _IFS_46_DAY_ARCHIVER_FIRE,
            timedelta(minutes=4),
        ),
    ],
)
def test_validation_monitor_window_spans_trigger_through_backstop_run(
    dataset_id: str,
    trigger_lower_bound: pd.Timestamp,
    typical_fire_to_validation: timedelta,
) -> None:
    dataset = _dataset(dataset_id)
    assert dataset.validation_monitor_schedule is not None
    assert dataset.validation_monitor_margin is not None
    update_cron = _update_cron(dataset)
    slot = update_cron.model_copy(
        update={"schedule": dataset.validation_monitor_schedule}
    ).next_fire_time(_DAY)
    backstop = update_cron.next_fire_time(slot)

    assert slot <= trigger_lower_bound
    assert backstop - slot < timedelta(days=1)
    assert slot + dataset.validation_monitor_margin >= (
        backstop + 2 * typical_fire_to_validation + _FIRST_REPASS_BACKOFF + _SLACK
    )
