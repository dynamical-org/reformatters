import time
from collections.abc import Callable

import numpy as np
import pandas as pd

_MAX_BACKOFF_SECONDS = 16.0


class DeadlineExceededError(TimeoutError):
    pass


def check_deadline(deadline: pd.Timestamp | None) -> None:
    if deadline is not None and pd.Timestamp.now() >= deadline:
        raise DeadlineExceededError(f"Deadline {deadline} reached")


def exponential_backoff_time(
    attempt: int, *, max_delay_seconds: float = _MAX_BACKOFF_SECONDS
) -> float:
    """Jittered exponential delay after a zero-indexed failed attempt."""
    max_delay = 0.2 if attempt == 0 else min(2 ** (attempt - 1), max_delay_seconds)
    return float(np.random.default_rng().uniform(0.5 * max_delay, max_delay))


def retry[T](
    func: Callable[[], T],
    max_attempts: int = 6,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    *,
    deadline: pd.Timestamp | None = None,
) -> T:
    """Retry utility that backs off exponentially between attempts."""
    last_exception = None
    for attempt in range(max_attempts):
        check_deadline(deadline)
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:  # sleep unless we're out of attempts
                delay = exponential_backoff_time(attempt)
                if deadline is not None:
                    delay = min(
                        delay, max(0.0, (deadline - pd.Timestamp.now()).total_seconds())
                    )
                time.sleep(delay)

    raise last_exception or AssertionError("unreachable")
