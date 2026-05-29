from unittest.mock import Mock

import pytest

from reformatters.common.retry import constant_jitter_delay, retry


def test_retry_succeeds_on_first_attempt() -> None:
    mock_func = Mock(return_value="success")
    result = retry(mock_func)
    assert result == "success"


def test_retry_succeeds_after_failures() -> None:
    mock_func = Mock(side_effect=[ValueError("fail"), "success"])
    result = retry(mock_func, max_attempts=3)
    assert result == "success"


def test_retry_fails_after_max_attempts() -> None:
    mock_func = Mock(side_effect=ValueError("persistent failure"))
    with pytest.raises(ValueError, match="persistent failure"):
        retry(mock_func, max_attempts=2)


def test_retryable_exceptions_retries_matching() -> None:
    mock_func = Mock(side_effect=[ValueError("transient"), "success"])
    result = retry(mock_func, max_attempts=3, retryable_exceptions=(ValueError,))
    assert result == "success"
    assert mock_func.call_count == 2


def test_retryable_exceptions_propagates_non_matching() -> None:
    mock_func = Mock(side_effect=TypeError("not retryable"))
    with pytest.raises(TypeError, match="not retryable"):
        retry(mock_func, max_attempts=3, retryable_exceptions=(ValueError,))
    assert mock_func.call_count == 1


def test_retry_uses_custom_delay_seconds() -> None:
    attempts_seen: list[int] = []

    def delay(attempt: int) -> float:
        attempts_seen.append(attempt)
        return 0.0

    mock_func = Mock(side_effect=[ValueError("a"), ValueError("b"), "success"])
    result = retry(mock_func, max_attempts=3, delay_seconds=delay)
    assert result == "success"
    assert attempts_seen == [0, 1]


def test_constant_jitter_delay_does_not_grow() -> None:
    delays = [constant_jitter_delay(attempt) for attempt in range(100)]
    assert all(0.05 <= d <= 0.2 for d in delays)
