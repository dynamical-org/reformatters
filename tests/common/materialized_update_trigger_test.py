from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError as KubernetesHTTPError

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import materialized_update_trigger as triggers
from reformatters.common.config import Config, Env
from reformatters.common.kubernetes import ReformatCronJob
from reformatters.common.source_availability import SourceAvailability, Summary
from reformatters.common.virtual_region_job import VirtualRegionJob
from reformatters.noaa.gefs.forecast_35_day_0_5_degree_virtual.region_job import (
    NoaaGefsForecast35Day05DegreeVirtualRegionJob,
)
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    SOURCE_AVAILABILITY,
)
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)


@pytest.fixture
def trigger() -> triggers.MaterializedUpdateTrigger:
    return triggers.MaterializedUpdateTrigger(
        source_cronjob="forecast-virtual-update",
        target_cronjob="forecast-update",
        availability=SourceAvailability(product_id="source", lead_hours=384),
    )


@pytest.fixture(autouse=True)
def no_running_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        triggers.kubernetes, "has_running_cronjob_job", Mock(return_value=False)
    )


@pytest.mark.parametrize(
    ("env", "cron", "enabled"),
    [
        (Env.test, "forecast-virtual-update", False),
        (Env.prod, "forecast-virtual-v2-update", False),
        (Env.prod, "forecast-virtual-update", True),
    ],
)
def test_trigger_is_production_only(
    monkeypatch: pytest.MonkeyPatch,
    trigger: triggers.MaterializedUpdateTrigger,
    env: Env,
    cron: str,
    enabled: bool,
) -> None:
    monkeypatch.setattr(Config, "env", env)
    monkeypatch.setenv("CRON_JOB_NAME", cron)
    assert (trigger.poller(pd.Timestamp("2026-10-07")) is not None) == enabled


def test_waits_for_summary_then_submits_once(
    monkeypatch: pytest.MonkeyPatch,
    trigger: triggers.MaterializedUpdateTrigger,
) -> None:
    init = pd.Timestamp("2020-01-01")
    fetch = Mock(return_value=Summary(products=[]))
    monkeypatch.setattr(Summary, "fetch", fetch)
    available = Mock(side_effect=[set(), {init}])
    monkeypatch.setattr(SourceAvailability, "available_inits", available)
    submit = Mock()
    monkeypatch.setattr(triggers.kubernetes, "create_job_from_cronjob", submit)
    clock = Mock(return_value=1)
    monkeypatch.setattr(triggers.time, "monotonic", clock)
    poller = triggers.TriggerPoller(trigger, init)
    assert poller.pending()
    assert poller.pending()
    assert fetch.call_count == 1
    submit.assert_not_called()
    clock.return_value = 61
    assert not poller.pending()
    assert not poller.pending()
    submit.assert_called_once_with("forecast-update", "forecast-t20010100-384h")


def test_gefs_waits_for_extension_age(monkeypatch: pytest.MonkeyPatch) -> None:
    trigger = (
        NoaaGefsForecast35Day05DegreeVirtualRegionJob.materialized_update_triggers[0]
    )
    assert trigger is not None
    assert trigger.availability.lead_hours == 840
    assert trigger.init_offset == pd.Timedelta("1D")
    assert trigger.min_init_age == pd.Timedelta("28h")
    current = (
        NoaaGefsForecast35Day05DegreeVirtualRegionJob.materialized_update_triggers[1]
    )
    assert current.availability.lead_hours == 384
    assert current.init_offset == pd.Timedelta(0)
    fetch = Mock()
    monkeypatch.setattr(Summary, "fetch", fetch)
    poller = triggers.TriggerPoller(trigger, pd.Timestamp.now() - pd.Timedelta("27h"))
    assert poller.pending()
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("timeout"), ValueError("bad summary")]
)
def test_summary_failure_defers_only_the_trigger(
    monkeypatch: pytest.MonkeyPatch,
    trigger: triggers.MaterializedUpdateTrigger,
    error: Exception,
) -> None:
    fetch = Mock(side_effect=error)
    monkeypatch.setattr(Summary, "fetch", fetch)
    submit = Mock()
    monkeypatch.setattr(triggers.kubernetes, "create_job_from_cronjob", submit)
    poller = triggers.TriggerPoller(trigger, pd.Timestamp("2020-01-01"))
    assert poller.pending()
    assert poller.pending()
    fetch.assert_called_once()
    submit.assert_not_called()


def test_ready_trigger_waits_for_busy_cron_to_finish(
    monkeypatch: pytest.MonkeyPatch,
    trigger: triggers.MaterializedUpdateTrigger,
) -> None:
    init = pd.Timestamp("2020-01-01")
    monkeypatch.setattr(Summary, "fetch", Mock(return_value=Summary(products=[])))
    monkeypatch.setattr(
        SourceAvailability, "available_inits", Mock(return_value={init})
    )
    busy = Mock(side_effect=[True, False])
    monkeypatch.setattr(triggers.kubernetes, "has_running_cronjob_job", busy)
    submit = Mock()
    monkeypatch.setattr(triggers.kubernetes, "create_job_from_cronjob", submit)
    clock = Mock(return_value=1)
    monkeypatch.setattr(triggers.time, "monotonic", clock)
    poller = triggers.TriggerPoller(trigger, init)
    assert poller.pending()
    submit.assert_not_called()
    assert not poller.submitted
    clock.return_value = 61
    assert not poller.pending()
    submit.assert_called_once()
    assert busy.call_args.args == ("forecast-update",)


@pytest.mark.parametrize(
    "error", [ApiException(status=503), KubernetesHTTPError("timeout")]
)
def test_kubernetes_failure_defers_only_the_trigger(
    monkeypatch: pytest.MonkeyPatch,
    trigger: triggers.MaterializedUpdateTrigger,
    error: Exception,
) -> None:
    init = pd.Timestamp("2020-01-01")
    monkeypatch.setattr(Summary, "fetch", Mock(return_value=Summary(products=[])))
    monkeypatch.setattr(
        SourceAvailability, "available_inits", Mock(return_value={init})
    )
    submit = Mock(side_effect=error)
    monkeypatch.setattr(triggers.kubernetes, "create_job_from_cronjob", submit)
    poller = triggers.TriggerPoller(trigger, init)
    assert poller.pending()
    assert not poller.submitted
    submit.assert_called_once()


def test_trigger_mappings_match_deployed_cronjobs_and_target_readiness() -> None:
    targets = {
        dataset._operational_cron_job(ReformatCronJob).name: dataset
        for dataset in DYNAMICAL_DATASETS
        if dataset.materialized_source_availability is not None
    }
    sources = [
        dataset
        for dataset in DYNAMICAL_DATASETS
        if issubclass(dataset.region_job_class, VirtualRegionJob)
        and dataset.region_job_class.materialized_update_triggers
    ]
    assert len(sources) == len(targets) == 4
    for dataset in sources:
        assert issubclass(dataset.region_job_class, VirtualRegionJob)
        for trigger in dataset.region_job_class.materialized_update_triggers:
            assert (
                trigger.source_cronjob
                == dataset._operational_cron_job(ReformatCronJob).name
            )
            target = targets[trigger.target_cronjob]
            assert "analysis" not in target.dataset_id
            availability = target.materialized_source_availability
            assert availability is not None
            assert trigger.availability.product_id == availability.product_id
            assert trigger.availability.components == availability.components
            if trigger.init_offset:
                assert target.reprocess_materialized_frontier
                assert trigger.init_offset == pd.Timedelta("1D")
                assert (availability.lead_hours, trigger.availability.lead_hours) == (
                    384,
                    840,
                )
            else:
                assert trigger.availability.lead_hours == availability.lead_hours


def test_hrrr_readiness_covers_every_materialized_source_family() -> None:
    config = NoaaHrrrForecast48HourTemplateConfig()
    assert {
        f"conus/{var.internal_attrs.hrrr_file_type}" for var in config.data_vars
    } == set(SOURCE_AVAILABILITY.components)
