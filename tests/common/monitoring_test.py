from datetime import timedelta
from unittest.mock import Mock

import pytest
import sentry_sdk.crons

from reformatters.common.config import Config
from reformatters.common.kubernetes import CronJob
from reformatters.common.monitoring import monitor_cron

_CRON_JOB = CronJob(
    command=["archive-grib-files"],
    workers_total=1,
    parallelism=1,
    name="example-archive-grib-files",
    schedule="0 4 * * *",
    pod_active_deadline=timedelta(hours=2),
    image="test-image:tag",
    dataset_id="example",
    cpu="1",
    memory="1G",
)


def test_monitor_cron_success_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(type(Config), "is_sentry_enabled", True)
    mock_capture = Mock()
    monkeypatch.setattr(sentry_sdk.crons, "capture_checkin", mock_capture)

    with monitor_cron(_CRON_JOB, "job-name"):
        pass
    statuses = [c.kwargs["status"] for c in mock_capture.call_args_list]
    assert statuses == ["in_progress", "ok"]

    call_kwargs = mock_capture.call_args_list[0].kwargs
    assert call_kwargs["monitor_config"]["schedule"]["value"] == "0 4 * * *"
    assert call_kwargs["monitor_config"]["max_runtime"] == 120

    mock_capture.reset_mock()
    with pytest.raises(ValueError, match="failure"):  # noqa: SIM117
        with monitor_cron(_CRON_JOB, "job-name"):
            raise ValueError("failure")
    statuses = [c.kwargs["status"] for c in mock_capture.call_args_list]
    assert statuses == ["in_progress", "error"]


def test_monitor_cron_without_sentry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(type(Config), "is_sentry_enabled", False)
    mock_capture = Mock()
    monkeypatch.setattr(sentry_sdk.crons, "capture_checkin", mock_capture)

    # Should not raise, and should not call out to sentry at all.
    with monitor_cron(_CRON_JOB, "job-name"):
        pass
    mock_capture.assert_not_called()
