from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

import pytest

from reformatters.dwd.archive_gribs.rclone_copyurl import (
    _tidy_stats,
    run_rclone_copyurl,
)


def test_tidy_stats_valid() -> None:
    line = "2026/01/31 16:15:41 ERROR :    16.342 MiB / 18.818 MiB, 87%, 0 B/s, ETA -"
    expected = "Transferred so far: 16.342 MiB. Recent throughput: 0 B/s"
    assert _tidy_stats(line) == expected


def test_tidy_stats_invalid() -> None:
    with pytest.raises(ValueError, match="Expected a colon"):
        _tidy_stats("some random log line")

    with pytest.raises(ValueError, match="Expected 4 comma-separated values"):
        _tidy_stats("2026/01/31 16:15:41 ERROR : only, three, values")


@patch(
    "reformatters.dwd.archive_gribs.rclone_copyurl._run_command_with_concurrent_logging"
)
@patch("pathlib.Path.write_text")
@patch("pathlib.Path.unlink")
def test_run_rclone_copyurl(
    mock_unlink: MagicMock, mock_write_text: MagicMock, mock_run_cmd: MagicMock
) -> None:
    csv_content = "src1,dst1\nsrc2,dst2"
    dst_root = PurePosixPath("/dst")

    run_rclone_copyurl(
        csv_of_files_to_transfer=csv_content,
        dst_root_path=dst_root,
        transfer_parallelism=4,
        checkers=4,
        stats_logging_freq="1m",
    )

    mock_write_text.assert_called_once_with(csv_content)
    mock_run_cmd.assert_called_once()
    mock_unlink.assert_called_once()

    cmd = mock_run_cmd.call_args[0][0]
    assert "rclone" in cmd
    assert "copyurl" in cmd
    assert "--transfers=4" in cmd
    assert "--checkers=4" in cmd
    assert "--stats=1m" in cmd
