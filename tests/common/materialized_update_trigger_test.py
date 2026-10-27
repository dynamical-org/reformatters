import re
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import kubernetes
from reformatters.common.materialized_update_trigger import MaterializedUpdateTrigger
from reformatters.noaa.hrrr.forecast_48_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualDataset,
)


def test_trigger_names_are_deterministic_per_consumer_and_unit() -> None:
    trigger = NoaaHrrrForecast48HourVirtualDataset.materialized_update_triggers[0]
    unit = pd.Timestamp("2026-10-27T00:00")
    name = trigger.job_name(unit)
    assert name == "noaa-hrrr-forecast-48-hour-t202610270000z"
    assert name == trigger.job_name(unit)
    assert name != trigger.job_name(unit + pd.Timedelta("6h"))
    assert name != trigger.job_name(unit + pd.Timedelta("30m"))
    other = MaterializedUpdateTrigger(target_cronjob="other-update")
    assert name != other.job_name(unit)
    assert not re.search(r"-\d+$", name)
    for retry in range(1, 17):
        allowed, retry_name = kubernetes.retry_job_name(name, max_retries=16)
        assert allowed
        assert len(retry_name) <= 52
        assert retry_name.endswith(f"-r{retry}")
        name = retry_name
    assert not kubernetes.retry_job_name(name, max_retries=16)[0]


def test_six_hourly_producer_triggers_daily_consumer_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trigger = MaterializedUpdateTrigger(target_cronjob="daily-update", init_hours=(0,))
    submit = Mock()
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", submit)
    for hour in (0, 6, 12, 18):
        trigger.dispatch(pd.Timestamp("2026-10-27") + pd.Timedelta(hours=hour))
    submit.assert_called_once_with(
        "daily-update", "daily-t202610270000z", kind="trigger"
    )


def test_hrrr_trigger_matches_live_consumer_cycles() -> None:
    (trigger,) = NoaaHrrrForecast48HourVirtualDataset.materialized_update_triggers
    consumer = next(
        dataset
        for dataset in DYNAMICAL_DATASETS
        if dataset.dataset_id == "noaa-hrrr-forecast-48-hour"
    )
    (cron,) = consumer.operational_kubernetes_resources("test")
    assert trigger.target_cronjob == cron.name
    config = consumer.template_config
    assert trigger.init_hours == (0, 6, 12, 18)
    positions = pd.date_range(
        config.append_dim_start, periods=4, freq=config.append_dim_frequency
    )
    assert {position.hour for position in positions} == set(trigger.init_hours)
    assert config.append_dim_frequency == pd.Timedelta("6h")


def test_long_consumer_names_keep_identity_within_indexed_job_budget() -> None:
    first = MaterializedUpdateTrigger(target_cronjob="forecast-" + "a" * 38 + "-update")
    second = MaterializedUpdateTrigger(
        target_cronjob="forecast-" + "a" * 37 + "b-update"
    )
    unit = pd.Timestamp("2026-10-27")
    name = first.job_name(unit)
    assert name == first.job_name(unit)
    assert name != second.job_name(unit)
    assert len(name) <= 52
    assert not re.search(r"-\d+$", name)
    assert len(kubernetes.retry_job_name(name, max_retries=16)[1]) <= 52
