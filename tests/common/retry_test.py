from unittest.mock import Mock

import pandas as pd
import pytest

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


@pytest.mark.parametrize("error", [ValueError, AssertionError])
def test_retry_until_returns_result_and_logs_each_retried_error(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], error: type[Exception]
) -> None:
    now = pd.Timestamp("2024-01-01")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: now))
    log = Mock()
    monkeypatch.setattr(retry_module, "log", log)
    operation = Mock(side_effect=[error(), error(), "recovered"])

    result = retry_module.retry_until(
        operation, now + pd.Timedelta(minutes=1), max_backoff_seconds=60
    )

    assert result == "recovered"
    assert log.exception.call_count == 2
    first, second = sleeps
    assert 0.1 <= first <= 0.2
    assert 0.5 <= second <= 1.0


def test_retry_until_raises_last_error_when_backoff_would_reach_deadline(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    now = pd.Timestamp("2024-01-01")
    monkeypatch.setattr(retry_module, "exponential_backoff_time", lambda *args: 1.0)
    last_error = ConnectionResetError()
    operation = Mock(side_effect=[ValueError(), last_error, "too late"])
    clock = Mock(side_effect=[now, now + pd.Timedelta(seconds=1)])
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: clock()))

    with pytest.raises(ConnectionResetError) as raised:
        retry_module.retry_until(
            operation, now + pd.Timedelta(seconds=2), max_backoff_seconds=60
        )

    assert raised.value is last_error
    assert operation.call_count == 2
    assert sleeps == [1.0]


def test_exponential_backoff_time_honors_custom_cap_for_large_attempts() -> None:
    assert 30 <= exponential_backoff_time(10_000, 60) <= 60
