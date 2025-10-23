import time
from collections.abc import Callable

import numpy as np

_rng = np.random.default_rng()


def retry[T](func: Callable[[], T], max_attempts: int = 6) -> T:
    """Simple retry utility that sleeps for a short time between attempts."""
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:  # noqa: BLE001
            last_exception = e
            if attempt < max_attempts - 1:  # sleep unless we're out of attempts
                time.sleep(attempt * _rng.uniform(0.8, 1.2) + 0.1)

    raise last_exception if last_exception else AssertionError("unreachable")
