import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from reformatters.common.logging import get_logger

_MAX_BACKOFF_SECONDS = 16.0
log = get_logger(__name__)


def exponential_backoff_time(
    attempt: int, max_backoff_seconds: float = _MAX_BACKOFF_SECONDS
) -> float:
    """Seconds to sleep after the given 0-indexed failed attempt: 0.2s, then 1s
    doubling to the cap, each jittered down to as little as half its value."""
    max_delay = min(0.2 if attempt == 0 else 2 ** (attempt - 1), max_backoff_seconds)
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


def retry_until[T](
    func: Callable[[], T], deadline: pd.Timestamp, *, max_backoff_seconds: float
) -> T:
    attempt = 0
    while True:
        try:
            return func()
        except Exception:
            delay = exponential_backoff_time(attempt, max_backoff_seconds)
            if pd.Timestamp.now() + pd.Timedelta(seconds=delay) >= deadline:
                raise
            log.exception("Attempt failed, retrying in %.1f seconds", delay)
            time.sleep(delay)
            attempt += 1
