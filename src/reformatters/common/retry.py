import time
from collections.abc import Callable

import numpy as np

_MAX_BACKOFF_SECONDS = 16.0


def exponential_backoff_time(attempt: int) -> float:
    """Seconds to sleep after the given 0-indexed failed attempt: 0.2s, then 1s
    doubling to a 16s cap, each jittered down to as little as half its value."""
    max_delay = 0.2 if attempt == 0 else min(2.0 ** (attempt - 1), _MAX_BACKOFF_SECONDS)
    return float(np.random.default_rng().uniform(0.5 * max_delay, max_delay))


def retry[T](
    func: Callable[[], T],
    max_attempts: int = 6,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Retry utility that backs off exponentially between attempts."""
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:  # sleep unless we're out of attempts
                time.sleep(exponential_backoff_time(attempt))

    raise last_exception or AssertionError("unreachable")
