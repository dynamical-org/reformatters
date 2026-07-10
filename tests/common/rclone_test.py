import pytest

from reformatters.common.rclone import _tidy_stats


def test_tidy_stats_valid() -> None:
    line = "2026/01/31 16:15:41 ERROR :    16.342 MiB / 18.818 MiB, 87%, 0 B/s, ETA -"
    expected = "Transferred so far: 16.342 MiB. Recent throughput: 0 B/s"
    assert _tidy_stats(line) == expected


def test_tidy_stats_invalid() -> None:
    with pytest.raises(ValueError, match="Expected a colon"):
        _tidy_stats("some random log line")

    with pytest.raises(ValueError, match="Expected 4 comma-separated values"):
        _tidy_stats("2026/01/31 16:15:41 ERROR : only, three, values")
