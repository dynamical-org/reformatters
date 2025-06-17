import pandas as pd


def whole_hours(timedelta: pd.Timedelta) -> int:
    """Assert the timedelta is a whole number of hours and return the int of hours."""
    assert timedelta.total_seconds() % 3600 == 0, (
        f"Timedelta {timedelta} is not a whole number of hours"
    )
    return int(timedelta.total_seconds() / 3600)
