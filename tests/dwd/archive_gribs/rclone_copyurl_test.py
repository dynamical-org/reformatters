from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

from reformatters.dwd.archive_gribs.rclone_copyurl import run_rclone_copyurl


@patch(
    "reformatters.dwd.archive_gribs.rclone_copyurl.run_command_with_concurrent_logging"
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
    assert "/usr/bin/rclone" in cmd
    assert "copyurl" in cmd
    assert "--transfers=4" in cmd
    assert "--checkers=4" in cmd
    assert "--stats=1m" in cmd
