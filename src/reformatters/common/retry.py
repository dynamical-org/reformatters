import time
from collections.abc import Callable

import numpy as np

from reformatters.common.logging import get_logger

log = get_logger(__name__)


def retry[T](
    func: Callable[[], T],
    max_attempts: int = 6,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Simple retry utility that sleeps for a short time between attempts."""
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:  # sleep unless we're out of attempts
                log.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed, retrying: "
                    f"{type(e).__name__}: {str(e)[:1000]}"
                )
                rng = np.random.default_rng()
                time.sleep(attempt * rng.uniform(0.8, 1.2) + 0.1)

    raise last_exception or AssertionError("unreachable")
