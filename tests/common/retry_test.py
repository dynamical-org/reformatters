from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.common import retry as retry_module
from reformatters.common.retry import exponential_backoff_time, retry


def test_deadline_prevents_retrying_an_inflight_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-10-20 19:07:59")
    deadline = pd.Timestamp("2026-10-20 19:08:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: now))
    calls = 0

    def operation() -> None:
        nonlocal now, calls
        calls += 1
        now = deadline
        raise OSError("transient storage failure")

    monkeypatch.setattr(retry_module.time, "sleep", lambda seconds: None)
    with pytest.raises(retry_module.DeadlineExceededError):
        retry(operation, max_attempts=10, deadline=deadline)
    assert calls == 1


def test_expired_deadline_starts_no_operation() -> None:
    operation = Mock()
    with pytest.raises(retry_module.DeadlineExceededError):
        retry(operation, deadline=pd.Timestamp("2000-01-01"))
    operation.assert_not_called()


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


def test_exponential_backoff_time_grows_then_caps() -> None:
    max_delays = [0.2, 1, 2, 4, 8, 16, 16, 16]
    for attempt, max_delay in enumerate(max_delays):
        # Jitter takes each delay down to as little as half its maximum.
        times = [exponential_backoff_time(attempt) for _ in range(50)]
        assert all(0.5 * max_delay <= t <= max_delay for t in times)
        assert min(times) < 0.75 * max_delay < max(times)

    # Long enough in total to ride out a transient object store outage.
    assert sum(exponential_backoff_time(a) for a in range(9)) > 35
