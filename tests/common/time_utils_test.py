import pandas as pd
import pytest

from reformatters.common.time_utils import whole_hours


class TestWholeHours:
    def test_one_hour(self) -> None:
        assert whole_hours(pd.Timedelta(hours=1)) == 1

    def test_multiple_hours(self) -> None:
        assert whole_hours(pd.Timedelta(hours=24)) == 24

    def test_zero_hours(self) -> None:
        assert whole_hours(pd.Timedelta(hours=0)) == 0

    def test_negative_hours(self) -> None:
        assert whole_hours(pd.Timedelta(hours=-3)) == -3

    def test_non_whole_hours_raises(self) -> None:
        with pytest.raises(AssertionError, match="not a whole number of hours"):
            whole_hours(pd.Timedelta(minutes=90))

    def test_fractional_seconds_raises(self) -> None:
        with pytest.raises(AssertionError, match="not a whole number of hours"):
            whole_hours(pd.Timedelta(hours=1, seconds=1))
