from unittest.mock import MagicMock, patch

import pytest

from reformatters.common.rclone import _tidy_stats, list_file_sizes


@patch("subprocess.run")
def test_list_file_sizes(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        stdout="123\thrrr.t12z.wrfsfcf00.grib2\n0\tempty.grib2\n",
        stderr="",
    )

    result = list_file_sizes(":s3:cache/prefix", env_vars={})

    assert result == {
        "hrrr.t12z.wrfsfcf00.grib2": 123,
        "empty.grib2": 0,
    }
    cmd = mock_run.call_args.args[0]
    assert cmd[:3] == ("/usr/bin/rclone", "lsf", ":s3:cache/prefix")
    assert "--format=sp" in cmd
    assert "--separator=\t" in cmd


def test_tidy_stats_valid() -> None:
    line = "2026/01/31 16:15:41 ERROR :    16.342 MiB / 18.818 MiB, 87%, 0 B/s, ETA -"
    expected = "Transferred so far: 16.342 MiB. Recent throughput: 0 B/s"
    assert _tidy_stats(line) == expected


def test_tidy_stats_invalid() -> None:
    with pytest.raises(ValueError, match="Expected a colon"):
        _tidy_stats("some random log line")

    with pytest.raises(ValueError, match="Expected 4 comma-separated values"):
        _tidy_stats("2026/01/31 16:15:41 ERROR : only, three, values")
