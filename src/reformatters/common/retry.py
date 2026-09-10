import time
from collections.abc import Callable

import numpy as np

_INIT_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 16.0


def sleep_before_retry(attempt: int) -> None:
    """Sleep after the given 0-indexed failed attempt, backing off exponentially with jitter."""
    backoff = min(_INIT_BACKOFF_SECONDS * 2**attempt, _MAX_BACKOFF_SECONDS)
    time.sleep(backoff * np.random.default_rng().uniform(0.8, 1.2))


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
                sleep_before_retry(attempt)

    raise last_exception or AssertionError("unreachable")
