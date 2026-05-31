import time
from collections.abc import Callable

import numpy as np


def linear_backoff_with_jitter(attempt: int) -> float:
    """Default delay: grows ~1s per attempt, with jitter."""
    return attempt * np.random.default_rng().uniform(0.8, 1.2) + 0.1


def constant_jitter_delay(_attempt: int) -> float:
    # Attempt-independent delay for optimistic-concurrency conflicts (e.g. an
    # icechunk amend losing a branch-tip CAS race): the contended resource
    # isn't overloaded, so backing off has no value. Jitter still avoids
    # concurrent workers retrying in lockstep.
    return np.random.default_rng().uniform(0.05, 0.2)


def retry[T](
    func: Callable[[], T],
    max_attempts: int = 6,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    delay_seconds: Callable[[int], float] = linear_backoff_with_jitter,
) -> T:
    """Simple retry utility that sleeps between attempts.

    delay_seconds maps the zero-based attempt number to the seconds to sleep
    before the next attempt.
    """
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:  # sleep unless we're out of attempts
                time.sleep(delay_seconds(attempt))

    raise last_exception or AssertionError("unreachable")
