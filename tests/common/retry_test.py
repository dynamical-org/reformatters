from unittest.mock import Mock

import pytest

from reformatters.common import retry as retry_module
from reformatters.common.retry import retry


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


def test_retry_backoff_grows_exponentially_then_caps(sleeps: list[float]) -> None:
    mock_func = Mock(side_effect=OSError("transient"))
    with pytest.raises(OSError, match="transient"):
        retry(mock_func, max_attempts=8)

    # Doubling from 1s, capped at 16s, each with +/-20% jitter.
    expected_unjittered = [1, 2, 4, 8, 16, 16, 16]
    assert len(sleeps) == len(expected_unjittered)
    for slept, expected in zip(sleeps, expected_unjittered, strict=True):
        assert 0.8 * expected <= slept <= 1.2 * expected
    # Long enough in total to ride out a transient object store outage.
    assert sum(sleeps) > 50
