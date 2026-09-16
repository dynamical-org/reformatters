import logging
import signal
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
import sentry_sdk
import sentry_sdk.crons
import typer
from typer.testing import CliRunner

from reformatters.__main__ import startup
from reformatters.common import monitoring
from reformatters.common.config import Config
from reformatters.common.kubernetes import CronJob
from reformatters.common.monitoring import (
    install_sigterm_logger,
    log_peak_memory,
    monitor_cron,
)

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


def test_log_cgroup_peak_memory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def read_text(path: Path) -> str:
        assert path == Path("/sys/fs/cgroup/memory.peak")
        return str(3 * 2**29) + "\n"

    monkeypatch.setattr(Path, "read_text", read_text)

    with caplog.at_level(logging.INFO):
        log_peak_memory()

    assert caplog.messages == [
        "Peak memory: 1.500 GiB (container cgroup, /sys/fs/cgroup/memory.peak)"
    ]


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError, OSError])
def test_log_peak_memory_unreadable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: type[OSError],
) -> None:
    read_text = Mock(side_effect=error)
    monkeypatch.setattr(Path, "read_text", read_text)

    with caplog.at_level(logging.INFO):
        log_peak_memory()

    assert caplog.messages == []
    read_text.assert_called_once_with()


@pytest.mark.parametrize(
    "outcome", ["success", "error", "exit", "interrupt", "sigterm"]
)
def test_cli_logs_peak_memory_on_exit(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    events = []
    monkeypatch.setattr(monitoring, "log_peak_memory", lambda: events.append("peak"))
    monkeypatch.setattr(sentry_sdk, "flush", Mock())
    app = typer.Typer(callback=startup)

    @app.command()
    def attempt() -> None:
        events.append("attempt")
        try:
            if outcome == "error":
                raise ValueError("job failed")
            if outcome == "exit":
                raise SystemExit(7)
            if outcome == "interrupt":
                raise KeyboardInterrupt
            if outcome == "sigterm":
                signal.raise_signal(signal.SIGTERM)
        finally:
            events.append("cleanup")

    original_handler = signal.getsignal(signal.SIGTERM)
    try:
        result = CliRunner().invoke(app, ["attempt"])
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    assert events == ["attempt", "cleanup", "peak"]
    assert (
        result.exit_code
        == {
            "success": 0,
            "error": 1,
            "exit": 7,
            "interrupt": 130,
            "sigterm": 143,
        }[outcome]
    )
    if outcome == "error":
        assert isinstance(result.exception, ValueError)
        assert str(result.exception) == "job failed"


def test_monitor_cron_success_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(type(Config), "is_sentry_enabled", True)
    mock_capture = Mock()
    mock_flush = Mock()
    monkeypatch.setattr(sentry_sdk.crons, "capture_checkin", mock_capture)
    monkeypatch.setattr(sentry_sdk, "flush", mock_flush)

    with monitor_cron(_CRON_JOB, "job-name"):
        pass
    statuses = [c.kwargs["status"] for c in mock_capture.call_args_list]
    assert statuses == ["in_progress", "ok"]

    call_kwargs = mock_capture.call_args_list[0].kwargs
    assert call_kwargs["monitor_config"]["schedule"]["value"] == "0 4 * * *"
    assert call_kwargs["monitor_config"]["max_runtime"] == 120
    mock_flush.assert_called_once_with(timeout=15)

    mock_capture.reset_mock()
    mock_flush.reset_mock()
    with pytest.raises(ValueError, match="failure"):  # noqa: SIM117
        with monitor_cron(_CRON_JOB, "job-name"):
            raise ValueError("failure")
    statuses = [c.kwargs["status"] for c in mock_capture.call_args_list]
    assert statuses == ["in_progress", "error"]
    mock_flush.assert_called_once_with(timeout=15)


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

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(SystemExit) as exit_info,
        ):
            signal.raise_signal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    assert "SIGTERM" in caplog.text
    assert mock_flush.call_count == 1
    assert exit_info.value.code == 128 + signal.SIGTERM
