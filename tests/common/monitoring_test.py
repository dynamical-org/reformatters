import logging
import os
import signal
from datetime import timedelta
from unittest.mock import Mock

import pytest
import sentry_sdk
import sentry_sdk.crons

from reformatters.common.config import Config
from reformatters.common.kubernetes import CronJob
from reformatters.common.monitoring import install_sigterm_logger, monitor_cron

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


def test_install_sigterm_logger(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original_handler = signal.getsignal(signal.SIGTERM)
    try:
        install_sigterm_logger()

        mock_flush = Mock()
        monkeypatch.setattr(sentry_sdk, "flush", mock_flush)
        exit_codes: list[int] = []
        monkeypatch.setattr(os, "_exit", exit_codes.append)

        with caplog.at_level(logging.ERROR):
            signal.raise_signal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    assert "SIGTERM" in caplog.text
    # Flushed before exiting, so the log reaches sentry.
    assert mock_flush.call_count == 1
    assert exit_codes == [128 + signal.SIGTERM]
