import time
from collections.abc import Callable
from typing import Any


def retry(func: Callable[[], Any], max_attempts: int = 6) -> Any:
    """Simple retry utility that sleeps for a short time between attempts."""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception:
            if attempt == max_attempts - 1:  # Last attempt failed
                raise
            time.sleep(attempt)
            continue
