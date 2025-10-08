import time
from collections.abc import Callable

import numpy as np


def retry[T](func: Callable[[], T], max_attempts: int = 6) -> T:
    """Simple retry utility that sleeps for a short time between attempts."""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception:
            if attempt == max_attempts - 1:  # Last attempt failed
                raise
            time.sleep(attempt * np.random.uniform(0.8, 1.2) + 0.1)
            continue

    raise AssertionError("Unreachable")
