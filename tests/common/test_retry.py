from unittest.mock import Mock

import pytest

from reformatters.common.retry import retry


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


def test_logs_each_retried_failure_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_func = Mock(side_effect=[ValueError("first"), ValueError("second"), "ok"])
    with caplog.at_level("WARNING"):
        retry(mock_func, max_attempts=4)

    assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]
    assert "Attempt 1/4 failed, retrying: ValueError: first" in caplog.text
    assert "Attempt 2/4 failed, retrying: ValueError: second" in caplog.text


def test_final_failure_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    # The last attempt raises to the caller, so logging it too would double report.
    mock_func = Mock(side_effect=ValueError("persistent"))
    with caplog.at_level("WARNING"), pytest.raises(ValueError, match="persistent"):
        retry(mock_func, max_attempts=2)

    assert len(caplog.records) == 1
    assert "Attempt 1/2" in caplog.text


def test_logged_exception_message_is_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_func = Mock(side_effect=[ValueError("x" * 5000), "ok"])
    with caplog.at_level("WARNING"):
        retry(mock_func, max_attempts=2)

    assert "x" * 1000 in caplog.text
    assert "x" * 1001 not in caplog.text
