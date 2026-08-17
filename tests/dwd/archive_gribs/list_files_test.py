from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

from reformatters.dwd.archive_gribs.list_files import (
    list_files_on_dst_for_all_nwp_runs_available_from_dwd,
    list_grib_files_on_dwd_https,
)


@patch("reformatters.dwd.archive_gribs.list_files.list_files")
def test_list_files_on_dst_for_all_nwp_runs(mock_list_files: MagicMock) -> None:
    src_paths = [
        PurePosixPath("alb_rad/file1_2026011400_000.grib2.bz2"),
        PurePosixPath("alb_rad/file2_2026011406_000.grib2.bz2"),
    ]

    # Mock list_files for each init datetime
    mock_list_files.side_effect = [
        [PurePosixPath("alb_rad/file1_2026011400_000.grib2.bz2")],  # for 2026-01-14T00
        [],  # for 2026-01-14T06
    ]

    result = list_files_on_dst_for_all_nwp_runs_available_from_dwd(
        src_paths_starting_with_nwp_var=src_paths,
        src_root_path_ending_with_init_hour=PurePosixPath("/src"),
        dst_root_path_without_init_dt=PurePosixPath("/dst"),
        checkers=4,
    )

    assert len(result) == 1
    assert (
        PurePosixPath("2026-01-14T00/alb_rad/file1_2026011400_000.grib2.bz2") in result
    )


@patch("reformatters.dwd.archive_gribs.list_files.list_files")
def test_list_grib_files_on_dwd_https(mock_list_files: MagicMock) -> None:
    list_grib_files_on_dwd_https(
        http_url="https://opendata.dwd.de", path="/path", checkers=4
    )

    mock_list_files.assert_called_once()
    kwargs = mock_list_files.call_args[1]
    assert kwargs["path"] == ":http:/path"
    assert "--http-url=https://opendata.dwd.de" in kwargs["rclone_args"]
