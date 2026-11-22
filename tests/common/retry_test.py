import logging
from collections.abc import Iterator
from datetime import timedelta
from itertools import pairwise
from unittest.mock import Mock

import icechunk
import pandas as pd
import pytest
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.transport import Transport

from reformatters.common import retry as retry_module
from reformatters.common.retry import exponential_backoff_time, retry


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", recorded.append)
    return recorded


def test_retry_succeeds_on_first_attempt(sleeps: list[float]) -> None:
    mock_func = Mock(return_value="success")
    result = retry(mock_func)
    assert result == "success"
    assert sleeps == []


def test_retry_succeeds_after_failures(sleeps: list[float]) -> None:
    mock_func = Mock(side_effect=[ValueError("fail"), "success"])
    result = retry(mock_func, max_attempts=3)
    assert result == "success"
    assert len(sleeps) == 1


def test_retry_fails_after_max_attempts(sleeps: list[float]) -> None:
    mock_func = Mock(side_effect=ValueError("persistent failure"))
    with pytest.raises(ValueError, match="persistent failure"):
        retry(mock_func, max_attempts=2)
    assert len(sleeps) == 1


def test_retryable_exceptions_retries_matching(sleeps: list[float]) -> None:
    mock_func = Mock(side_effect=[ValueError("transient"), "success"])
    result = retry(mock_func, max_attempts=3, retryable_exceptions=(ValueError,))
    assert result == "success"
    assert mock_func.call_count == 2


def test_retryable_exceptions_propagates_non_matching(sleeps: list[float]) -> None:
    mock_func = Mock(side_effect=TypeError("not retryable"))
    with pytest.raises(TypeError, match="not retryable"):
        retry(mock_func, max_attempts=3, retryable_exceptions=(ValueError,))
    assert mock_func.call_count == 1
    assert sleeps == []


def test_retry_sleeps_between_every_attempt(sleeps: list[float]) -> None:
    mock_func = Mock(side_effect=OSError("transient"))
    with pytest.raises(OSError, match="transient"):
        retry(mock_func, max_attempts=8)

    assert len(sleeps) == 7


def test_non_retryable_subclass_propagates_without_sleep(sleeps: list[float]) -> None:
    mock_func = Mock(side_effect=FileNotFoundError("permanent"))
    with pytest.raises(FileNotFoundError, match="permanent"):
        retry(
            mock_func,
            retryable_exceptions=(OSError,),
            non_retryable_exceptions=(FileNotFoundError,),
        )
    assert mock_func.call_count == 1
    assert sleeps == []


def test_exponential_backoff_time_grows_then_caps() -> None:
    max_delays = [0.2, 1, 2, 4, 8, 16, 16, 16]
    for attempt, max_delay in enumerate(max_delays):
        # Jitter takes each delay down to as little as half its maximum.
        times = [exponential_backoff_time(attempt) for _ in range(50)]
        assert all(0.5 * max_delay <= t <= max_delay for t in times)
        assert min(times) < 0.75 * max_delay < max(times)

    # Long enough in total to ride out a transient object store outage.
    assert sum(exponential_backoff_time(a) for a in range(9)) > 35


class FaultClock:
    now = pd.Timestamp("2024-01-01")

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


@pytest.fixture
def fault_clock(monkeypatch: pytest.MonkeyPatch) -> FaultClock:
    clock = FaultClock()
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda cls, tz=None: clock.now)
    )
    monkeypatch.setattr(retry_module.time, "sleep", clock.sleep)
    return clock


@pytest.fixture
def alert_times(
    fault_clock: FaultClock, caplog: pytest.LogCaptureFixture
) -> Iterator[list[pd.Timestamp]]:
    times: list[pd.Timestamp] = []

    class RecordAlertTime(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.levelno == logging.ERROR:
                times.append(fault_clock.now)
            return True

    with caplog.filtering(RecordAlertTime()):
        yield times


@pytest.mark.parametrize("seconds_until_deadline", [60, 0])
def test_virtual_update_returns_first_success_without_sleep_or_logs(
    fault_clock: FaultClock,
    caplog: pytest.LogCaptureFixture,
    seconds_until_deadline: int,
) -> None:
    result = object()
    attempt = Mock(return_value=result)
    assert (
        retry_module.retry_virtual_update(
            attempt,
            dataset_id="test-dataset",
            give_up_at=fault_clock.now + timedelta(seconds=seconds_until_deadline),
        )
        is result
    )
    attempt.assert_called_once()
    assert not fault_clock.sleeps
    assert not caplog.records


def test_unexpected_update_error_alerts_once_immediately_with_stack_and_fingerprint(
    fault_clock: FaultClock, caplog: pytest.LogCaptureFixture
) -> None:
    envelopes: list[Envelope] = []

    class RecordingTransport(Transport):
        def capture_envelope(self, envelope: Envelope) -> None:
            envelopes.append(envelope)

    client = sentry_sdk.Client(
        dsn="https://key@example.invalid/1",
        transport=RecordingTransport(),
        default_integrations=False,
        integrations=[LoggingIntegration()],
    )
    attempts = 0
    fault_clock.now = pd.Timestamp("2024-01-01T00:00:00.123456-05:00")
    start = fault_clock.now

    def attempt() -> str:
        nonlocal attempts
        assert len(envelopes) == (0, 1, 1, 2, 2)[attempts]
        attempts += 1
        if attempts == 3:
            fault_clock.now = start + timedelta(minutes=8)
        if attempts <= 4:
            raise RuntimeError("discovery failed")
        return "recovered"

    try:
        with sentry_sdk.isolation_scope() as isolation, sentry_sdk.new_scope() as scope:
            isolation.fingerprint = ["existing-fingerprint"]
            scope.set_client(client)
            assert (
                retry_module.retry_virtual_update(
                    attempt,
                    dataset_id="test-dataset",
                    give_up_at=start + timedelta(minutes=10),
                )
                == "recovered"
            )
            logging.getLogger(__name__).error("Unrelated failure")
    finally:
        client.close()

    first_alert, second_fault_alert, unrelated_alert = [
        payload
        for envelope in envelopes
        for item in envelope.items
        if item.type == "event" and (payload := item.payload.json) is not None
    ]
    assert first_alert["fingerprint"] == [
        "test-dataset",
        "RuntimeError",
        "2024-01-01T05:00:00Z",
    ]
    assert second_fault_alert["fingerprint"] == [
        "test-dataset",
        "RuntimeError",
        "2024-01-01T05:08:00Z",
    ]
    assert (
        first_alert["exception"]["values"][0]["stacktrace"]["frames"][-1]["function"]
        == "attempt"
    )
    assert unrelated_alert["fingerprint"] == ["existing-fingerprint"]
    first_message = next(
        r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
    )
    assert "2024-01-01T05:00:00Z" in first_message
    assert "1 failed attempt" in first_message
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 4
    assert all(
        "RuntimeError" in warning and "discovery failed" in warning
        for warning in warnings
    )


def test_new_update_error_type_alerts_on_first_occurrence_within_same_fault(
    fault_clock: FaultClock, caplog: pytest.LogCaptureFixture
) -> None:
    errors = [
        RuntimeError("first"),
        RuntimeError("first again"),
        OSError("second"),
        OSError("second again"),
    ]
    observed: list[list[type[BaseException]]] = []

    def attempt() -> str:
        observed.append(
            [
                r.exc_info[0]
                for r in caplog.records
                if r.levelno == logging.ERROR and r.exc_info and r.exc_info[0]
            ]
        )
        if errors:
            raise errors.pop(0)
        return "recovered"

    assert (
        retry_module.retry_virtual_update(
            attempt,
            dataset_id="test-dataset",
            give_up_at=fault_clock.now + timedelta(minutes=5),
        )
        == "recovered"
    )
    assert observed == [
        [],
        [RuntimeError],
        [RuntimeError],
        [RuntimeError, OSError],
        [RuntimeError, OSError],
    ]


@pytest.mark.parametrize(
    "error",
    [AssertionError(), SystemExit(), KeyboardInterrupt()],
    ids=["assertion", "system-exit", "keyboard-interrupt"],
)
def test_virtual_update_does_not_retry_invariant_or_termination_errors(
    fault_clock: FaultClock, error: BaseException
) -> None:
    attempt = Mock(side_effect=error)
    with pytest.raises(type(error)) as raised:
        retry_module.retry_virtual_update(
            attempt,
            dataset_id="test-dataset",
            give_up_at=fault_clock.now + timedelta(minutes=1),
        )
    assert raised.value is error
    attempt.assert_called_once()
    assert not fault_clock.sleeps


@pytest.mark.parametrize("remaining_seconds", [1.0, 0.5], ids=["at", "past"])
def test_virtual_retry_preserves_last_error_when_next_attempt_would_miss_deadline(
    monkeypatch: pytest.MonkeyPatch, fault_clock: FaultClock, remaining_seconds: float
) -> None:
    monkeypatch.setattr(retry_module, "exponential_backoff_time", lambda _: 1)
    first = icechunk.StorageError("first fault", "io")
    last = icechunk.StorageError("last fault", "io")
    attempt = Mock(side_effect=[first, last])
    with pytest.raises(icechunk.StorageError) as raised:
        retry_module.retry_virtual_update(
            attempt,
            dataset_id="test-dataset",
            give_up_at=fault_clock.now + timedelta(seconds=1 + remaining_seconds),
        )
    assert raised.value is last
    assert attempt.call_count == 2
    assert fault_clock.sleeps == [1]


def test_virtual_retry_does_not_start_another_attempt_after_oversleeping(
    monkeypatch: pytest.MonkeyPatch, fault_clock: FaultClock
) -> None:
    give_up_at = fault_clock.now + timedelta(seconds=1)
    monkeypatch.setattr(retry_module.time, "sleep", lambda _: fault_clock.advance(2))
    error = icechunk.StorageError("storage unavailable", "io")
    attempt = Mock(side_effect=error)
    with pytest.raises(icechunk.StorageError) as raised:
        retry_module.retry_virtual_update(
            attempt, dataset_id="test-dataset", give_up_at=give_up_at
        )
    assert raised.value is error
    assert fault_clock.now >= give_up_at
    attempt.assert_called_once()


@pytest.mark.parametrize("attempt_seconds", [0, 30], ids=["instant", "slow"])
def test_storage_fault_alerts_between_two_and_three_minutes_even_with_backoff(
    fault_clock: FaultClock,
    alert_times: list[pd.Timestamp],
    monkeypatch: pytest.MonkeyPatch,
    attempt_seconds: int,
) -> None:
    rng = Mock()
    rng.uniform.side_effect = lambda low, high: high
    monkeypatch.setattr(retry_module.np.random, "default_rng", lambda: rng)
    start = fault_clock.now

    def attempt() -> str:
        if fault_clock.now < start + timedelta(minutes=4):
            fault_clock.advance(attempt_seconds)
            raise icechunk.StorageError("commit timed out", "io")
        return "recovered"

    assert (
        retry_module.retry_virtual_update(
            attempt,
            dataset_id="test-dataset",
            give_up_at=start + timedelta(hours=1),
        )
        == "recovered"
    )
    (alert_at,) = alert_times
    assert timedelta(minutes=2) <= alert_at - start <= timedelta(minutes=3)


@pytest.mark.parametrize(
    (
        "first_fault_seconds",
        "second_fault_start_minute",
        "second_fault_seconds",
        "alert_seconds",
    ),
    [
        (220, 7, 220, [120]),
        (10, 8, 10, []),
        (10, 4, 10, [240]),
        (220, 10, 220, [120, 720]),
    ],
    ids=[
        "deduplicates-recent-fault",
        "brief-new-fault",
        "brief-same-fault",
        "sustained-new-fault",
    ],
)
def test_storage_blips_reset_alerts_after_five_minutes_and_backoff_after_one_minute(
    fault_clock: FaultClock,
    alert_times: list[pd.Timestamp],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    first_fault_seconds: int,
    second_fault_start_minute: int,
    second_fault_seconds: int,
    alert_seconds: list[int],
) -> None:
    rng = Mock()
    rng.uniform.side_effect = lambda low, high: low
    monkeypatch.setattr(retry_module.np.random, "default_rng", lambda: rng)
    start = fault_clock.now
    second_fault_start = start + timedelta(minutes=second_fault_start_minute)
    first_failures = 0
    second_fault_delays: list[float] = []

    def attempt() -> str:
        nonlocal first_failures
        if fault_clock.now < start + timedelta(seconds=first_fault_seconds):
            first_failures += 1
            raise icechunk.StorageError("first fault", "io")
        if fault_clock.now < second_fault_start:
            fault_clock.now = second_fault_start
            raise icechunk.StorageError("second fault", "io")
        second_fault_delays.append(
            (fault_clock.now - second_fault_start).total_seconds()
        )
        if fault_clock.now < second_fault_start + timedelta(
            seconds=second_fault_seconds
        ):
            raise icechunk.StorageError("second fault", "io")
        return "recovered"

    assert (
        retry_module.retry_virtual_update(
            attempt,
            dataset_id="test-dataset",
            give_up_at=second_fault_start + timedelta(minutes=6),
        )
        == "recovered"
    )
    assert alert_times == [start + timedelta(seconds=s) for s in alert_seconds]
    assert second_fault_delays[0] == 0.1
    if second_fault_seconds == 10:
        assert second_fault_delays[1] < second_fault_seconds
    if alert_seconds == [240]:
        (alert,) = [
            r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
        ]
        assert "2024-01-01T00:00:00Z" in alert
        assert f"{first_failures + 1} failed attempts" in alert


def test_virtual_retry_slows_after_five_failures_without_overflowing(
    fault_clock: FaultClock,
) -> None:
    starts: list[pd.Timestamp] = []

    def attempt() -> int:
        starts.append(fault_clock.now)
        if len(starts) <= 1100:
            raise icechunk.StorageError("persistent fault", "io")
        return len(starts)

    assert (
        retry_module.retry_virtual_update(
            attempt,
            dataset_id="test-dataset",
            give_up_at=fault_clock.now + timedelta(hours=24),
        )
        == 1101
    )
    delays = [(right - left).total_seconds() for left, right in pairwise(starts)]
    for delay, max_delay in zip(delays[:5], (0.2, 1, 2, 4, 8), strict=True):
        assert max_delay / 2 <= delay <= max_delay
    assert all(30 <= delay <= 60 for delay in delays[5:])
