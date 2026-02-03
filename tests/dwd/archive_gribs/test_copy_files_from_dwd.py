from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

import pytest

from reformatters.dwd.archive_gribs.copy_files_from_dwd import (
    compute_which_files_still_need_to_be_transferred,
    copy_files_from_dwd_https,
)


def test_compute_which_files_still_need_to_be_transferred() -> None:
    src_paths = [
        PurePosixPath("alb_rad/file1_2026011400_000.grib2.bz2"),
        PurePosixPath("alb_rad/file2_2026011400_000.grib2.bz2"),
    ]
    files_already_on_dst = {
        PurePosixPath("2026-01-14T00/alb_rad/file1_2026011400_000.grib2.bz2")
    }
    src_host_and_root_path = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00"

    result = compute_which_files_still_need_to_be_transferred(
        src_paths, files_already_on_dst, src_host_and_root_path
    )

    assert len(result) == 1
    expected_full_src = (
        f"{src_host_and_root_path}/alb_rad/file2_2026011400_000.grib2.bz2"
    )
    expected_dst = "2026-01-14T00/alb_rad/file2_2026011400_000.grib2.bz2"
    assert result[0] == f"{expected_full_src},{expected_dst}"


def test_copy_files_from_dwd_https_input_validation() -> None:
    with pytest.raises(ValueError, match="must start with a forward slash"):
        copy_files_from_dwd_https(
            src_host="https://opendata.dwd.de",
            src_root_path=PurePosixPath("relative/path"),
            dst_root_path=PurePosixPath("/dst"),
            transfer_parallelism=4,
            checkers=4,
            stats_logging_freq="1m",
        )


@patch(
    "reformatters.dwd.archive_gribs.copy_files_from_dwd.list_grib_files_on_dwd_https"
)
@patch(
    "reformatters.dwd.archive_gribs.copy_files_from_dwd.list_files_on_dst_for_all_nwp_runs_available_from_dwd"
)
@patch("reformatters.dwd.archive_gribs.copy_files_from_dwd.run_rclone_copyurl")
def test_copy_files_from_dwd_https_strips_trailing_slash(
    mock_run: MagicMock, mock_list_dst: MagicMock, mock_list_src: MagicMock
) -> None:
    mock_list_src.return_value = []
    mock_list_dst.return_value = set()

    copy_files_from_dwd_https(
        src_host="https://opendata.dwd.de/",
        src_root_path=PurePosixPath("/weather/nwp/icon-eu/grib/00/"),
        dst_root_path=PurePosixPath("/dst"),
        transfer_parallelism=4,
        checkers=4,
        stats_logging_freq="1m",
    )

    # Check that src_host in the call to list_grib_files_on_dwd_https was stripped
    mock_list_src.assert_called_once()
    assert mock_list_src.call_args[1]["http_url"] == "https://opendata.dwd.de"
