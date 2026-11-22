import time
from collections.abc import Callable
from datetime import timedelta

import icechunk
import numpy as np
import pandas as pd
import sentry_sdk

from reformatters.common.logging import get_logger
from reformatters.common.types import Timestamp

_MAX_BACKOFF_SECONDS = 16.0
_FAST_UPDATE_RETRIES = 5
_SLOW_UPDATE_RETRY_MIN_SECONDS = 30
_SLOW_UPDATE_RETRY_MAX_SECONDS = 60
_STORAGE_FAULT_ALERT_AFTER = timedelta(minutes=2)
_STORAGE_FAULT_RESET_AFTER = timedelta(minutes=15)

log = get_logger(__name__)


def exponential_backoff_time(attempt: int) -> float:
    """Seconds to sleep after the given 0-indexed failed attempt: 0.2s, then 1s
    doubling to a 16s cap, each jittered down to as little as half its value."""
    max_delay = 0.2 if attempt == 0 else min(2.0 ** (attempt - 1), _MAX_BACKOFF_SECONDS)
    return float(np.random.default_rng().uniform(0.5 * max_delay, max_delay))


def retry[T](
    func: Callable[[], T],
    max_attempts: int = 6,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    *,
    non_retryable_exceptions: tuple[type[Exception], ...] = (),
) -> T:
    """Retry utility that backs off exponentially between attempts."""
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except non_retryable_exceptions:
            raise
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:  # sleep unless we're out of attempts
                time.sleep(exponential_backoff_time(attempt))

    raise last_exception or AssertionError("unreachable")


def retry_virtual_update[T](
    attempt: Callable[[], T],
    *,
    dataset_id: str,
    give_up_at: Timestamp,
    non_retryable_exceptions: tuple[type[Exception], ...] = (),
) -> T:
    """Retry a virtual update with backoff while another attempt fits before its poll deadline."""
    fail_fast = (AssertionError, *non_retryable_exceptions)
    deadline_utc = (
        give_up_at.tz_convert(None) if give_up_at.tz is not None else give_up_at
    )
    fault_started_at = None
    alerted: set[type[Exception]] = set()
    retry_count = 0
    started_at = pd.Timestamp.now(tz=give_up_at.tz)

    while True:
        try:
            return attempt()
        except fail_fast:
            raise
        except Exception as error:
            failed_at = pd.Timestamp.now(tz=give_up_at.tz)
            duration = failed_at - started_at
            if fault_started_at is None or duration > _STORAGE_FAULT_RESET_AFTER:
                fault_started_at = failed_at
                alerted.clear()
                retry_count = 0

            delay = (
                exponential_backoff_time(retry_count)
                if retry_count < _FAST_UPDATE_RETRIES
                else float(
                    np.random.default_rng().uniform(
                        _SLOW_UPDATE_RETRY_MIN_SECONDS, _SLOW_UPDATE_RETRY_MAX_SECONDS
                    )
                )
            )
            next_attempt_at = failed_at + timedelta(seconds=delay)
            if next_attempt_at >= give_up_at:
                raise

            error_type = type(error)
            error_message = (
                error.message
                if isinstance(error, icechunk.StorageError)
                else str(error)
            )
            log.warning(
                "Retrying after %s (%s) in an attempt that ran for %.1f s",
                error_type.__name__,
                error_message,
                duration.total_seconds(),
            )
            alert_at = (
                fault_started_at + _STORAGE_FAULT_ALERT_AFTER
                if isinstance(error, icechunk.StorageError)
                else failed_at
            )
            if error_type not in alerted and alert_at <= next_attempt_at:
                if alert_at > failed_at:
                    time.sleep((alert_at - failed_at).total_seconds())
                now = pd.Timestamp.now(tz=give_up_at.tz)
                if now >= give_up_at:
                    raise
                with sentry_sdk.new_scope() as scope:
                    scope.fingerprint = [dataset_id, error_type.__name__]
                    log.exception(
                        "%s has been failing for %.1f minutes (%s: %s). "
                        "The update is still retrying until %sZ and needs no rerun unless the Job later fails.",
                        dataset_id,
                        (now - fault_started_at).total_seconds() / 60,
                        error_type.__name__,
                        error_message,
                        deadline_utc.isoformat(),
                    )
                alerted.add(error_type)

            remaining_sleep = (
                next_attempt_at - pd.Timestamp.now(tz=give_up_at.tz)
            ).total_seconds()
            if remaining_sleep > 0:
                time.sleep(remaining_sleep)
            started_at = pd.Timestamp.now(tz=give_up_at.tz)
            if started_at >= give_up_at:
                raise
            retry_count = min(retry_count + 1, _FAST_UPDATE_RETRIES)
