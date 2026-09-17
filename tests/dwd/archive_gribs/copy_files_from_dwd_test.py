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

    assert result == [
        (
            f"{src_host_and_root_path}/alb_rad/file2_2026011400_000.grib2.bz2",
            PurePosixPath("2026-01-14T00/alb_rad/file2_2026011400_000.grib2.bz2"),
        )
    ]


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
@patch("reformatters.dwd.archive_gribs.copy_files_from_dwd.copy_urls")
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


@patch(
    "reformatters.dwd.archive_gribs.copy_files_from_dwd.list_grib_files_on_dwd_https"
)
@patch(
    "reformatters.dwd.archive_gribs.copy_files_from_dwd.list_files_on_dst_for_all_nwp_runs_available_from_dwd"
)
@patch("reformatters.dwd.archive_gribs.copy_files_from_dwd.copy_urls")
def test_copy_files_from_dwd_https_hands_missing_files_to_copy_urls(
    mock_copy_urls: MagicMock, mock_list_dst: MagicMock, mock_list_src: MagicMock
) -> None:
    mock_list_src.return_value = [
        PurePosixPath("alb_rad/file1_2026011400_000.grib2.bz2"),
        PurePosixPath("alb_rad/file2_2026011400_000.grib2.bz2"),
    ]
    mock_list_dst.return_value = {
        PurePosixPath("2026-01-14T00/alb_rad/file1_2026011400_000.grib2.bz2")
    }
    env_vars = {"RCLONE_S3_PROVIDER": "AWS"}

    copy_files_from_dwd_https(
        src_host="https://opendata.dwd.de",
        src_root_path=PurePosixPath("/weather/nwp/icon-eu/grib/00"),
        dst_root_path=PurePosixPath(":s3:bucket/root"),
        transfer_parallelism=4,
        checkers=4,
        stats_logging_freq="1m",
        env_vars=env_vars,
    )

    mock_copy_urls.assert_called_once_with(
        sources_and_dst_paths=[
            (
                "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/alb_rad/file2_2026011400_000.grib2.bz2",
                PurePosixPath("2026-01-14T00/alb_rad/file2_2026011400_000.grib2.bz2"),
            )
        ],
        dst_root_path=":s3:bucket/root",
        transfer_parallelism=4,
        checkers=4,
        stats_logging_freq="1m",
        env_vars=env_vars,
    )


@patch(
    "reformatters.dwd.archive_gribs.copy_files_from_dwd.list_grib_files_on_dwd_https"
)
@patch(
    "reformatters.dwd.archive_gribs.copy_files_from_dwd.list_files_on_dst_for_all_nwp_runs_available_from_dwd"
)
@patch("reformatters.dwd.archive_gribs.copy_files_from_dwd.copy_urls")
def test_copy_files_from_dwd_https_retries_a_failed_copy_with_only_the_files_still_missing(
    mock_copy_urls: MagicMock, mock_list_dst: MagicMock, mock_list_src: MagicMock
) -> None:
    mock_list_src.return_value = [
        PurePosixPath("alb_rad/file1_2026011400_000.grib2.bz2"),
        PurePosixPath("alb_rad/file2_2026011400_000.grib2.bz2"),
    ]
    mock_list_dst.side_effect = [
        set(),
        {PurePosixPath("2026-01-14T00/alb_rad/file1_2026011400_000.grib2.bz2")},
    ]
    mock_copy_urls.side_effect = [
        RuntimeError("rclone copyurl exited with code 1"),
        None,
    ]

    with patch("reformatters.common.retry.time.sleep"):
        copy_files_from_dwd_https(
            src_host="https://opendata.dwd.de",
            src_root_path=PurePosixPath("/weather/nwp/icon-eu/grib/00"),
            dst_root_path=PurePosixPath(":s3:bucket/root"),
            transfer_parallelism=4,
            checkers=4,
            stats_logging_freq="1m",
        )

    assert mock_list_src.call_count == 1
    assert [
        [dst for _url, dst in c.kwargs["sources_and_dst_paths"]]
        for c in mock_copy_urls.call_args_list
    ] == [
        [
            PurePosixPath("2026-01-14T00/alb_rad/file1_2026011400_000.grib2.bz2"),
            PurePosixPath("2026-01-14T00/alb_rad/file2_2026011400_000.grib2.bz2"),
        ],
        [PurePosixPath("2026-01-14T00/alb_rad/file2_2026011400_000.grib2.bz2")],
    ]
