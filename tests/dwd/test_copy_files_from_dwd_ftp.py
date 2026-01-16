import json
from datetime import datetime
from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

import pytest

from reformatters.dwd.copy_files_from_dwd_ftp import (
    _compute_copy_plan,
    _copy_batches,
    _PathAndSize,
    copy_files_from_dwd_ftp,
    list_ftp_files,
)
from reformatters.dwd.parse_rclone_log import TransferSummary


@pytest.fixture
def mock_lsf_output() -> list[str]:
    return [
        "alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2,100",
        "alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_001_ALB_RAD.grib2.bz2,200",
        "t_2m/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_T_2M.grib2.bz2,300",
        "ignore_me/pressure_level.grib2,400",
    ]


def test_list_ftp_files(mock_lsf_output: list[str]) -> None:
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "\n".join(mock_lsf_output)
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = list_ftp_files("host", PurePosixPath("/path"))

        assert len(result) == 4
        assert result[0].path.name.startswith("icon-eu")
        assert result[0].size_bytes == 100
        assert isinstance(result[0], _PathAndSize)
        mock_run.assert_called_once()


def test_compute_copy_plan(mock_lsf_output: list[str]) -> None:
    file_list = [
        _PathAndSize(PurePosixPath(line.split(",")[0]), int(line.split(",")[1]))
        for line in mock_lsf_output
    ]

    plan = _compute_copy_plan(file_list=file_list)

    # Should have 2 batches keyed by (datetime, variable)
    dt = datetime(2026, 1, 14, 0)
    assert len(plan) == 2
    assert (dt, "alb_rad") in plan
    assert (dt, "t_2m") in plan
    assert len(plan[(dt, "alb_rad")]) == 2
    assert len(plan[(dt, "t_2m")]) == 1


def test_compute_copy_plan_with_limit(mock_lsf_output: list[str]) -> None:
    file_list = [
        _PathAndSize(PurePosixPath(line.split(",")[0]), int(line.split(",")[1]))
        for line in mock_lsf_output
    ]

    # Limit to 1 file per variable.
    plan = _compute_copy_plan(file_list=file_list, max_files_per_nwp_variable=1)

    dt = datetime(2026, 1, 14, 0)
    assert len(plan) == 2
    assert len(plan[(dt, "alb_rad")]) == 1
    assert len(plan[(dt, "t_2m")]) == 1
    # First alb_rad file should be there
    assert file_list[0] in plan[(dt, "alb_rad")]
    # Second alb_rad file should be skipped
    assert file_list[1] not in plan[(dt, "alb_rad")]
    # t_2m file should be there
    assert file_list[2] in plan[(dt, "t_2m")]


def test_copy_batches() -> None:
    ftp_path = PurePosixPath("/ftp")
    dst_root = PurePosixPath("/dst")
    dt = datetime(2026, 1, 14, 0)
    copy_plan = {
        (dt, "alb_rad"): [_PathAndSize(PurePosixPath("alb_rad/file1.bz2"), 100)],
        (dt, "t_2m"): [_PathAndSize(PurePosixPath("t_2m/file2.bz2"), 200)],
    }

    mock_json_logs = [
        {"level": "info", "msg": "Copied (new)", "object": "alb_rad/file1.bz2"},
        {
            "level": "info",
            "msg": "Summary stats",
            "stats": {
                "totalTransfers": 1,
                "totalChecks": 1,
                "errors": 0,
                "totalBytes": 100,
                "elapsedTime": 0.1,
                "transferTime": 0.1,
                "listed": 1,
            },
        },
    ]
    mock_stderr = "\n".join(json.dumps(entry) for entry in mock_json_logs)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = mock_stderr
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        summary = _copy_batches("host", ftp_path, dst_root, copy_plan)

        assert mock_run.call_count == 2
        assert isinstance(summary, TransferSummary)
        # 2 batches, each with 1 transfer and 1 check
        assert summary.total_transfers == 2
        assert summary.total_checks == 2
        assert summary.errors == 0
        assert summary.total_bytes == 200


def test_copy_files_from_dwd_ftp(mock_lsf_output: list[str]) -> None:
    ftp_host = "opendata.dwd.de"
    ftp_path = PurePosixPath("/weather/nwp/icon-eu/grib/00")
    dst_root = PurePosixPath("/dst")

    with patch("subprocess.run") as mock_run:
        mock_result_ls = MagicMock()
        mock_result_ls.stdout = "\n".join(mock_lsf_output)
        mock_result_ls.stderr = ""
        mock_result_ls.returncode = 0

        mock_result_copy = MagicMock()
        mock_result_copy.stdout = ""
        mock_result_copy.stderr = json.dumps(
            {
                "level": "info",
                "msg": "Summary stats",
                "stats": {
                    "totalTransfers": 1,
                    "totalChecks": 0,
                    "errors": 0,
                    "totalBytes": 50,
                    "elapsedTime": 0.1,
                    "transferTime": 0.1,
                    "listed": 0,
                },
            }
        )
        mock_result_copy.returncode = 0

        mock_run.side_effect = [mock_result_ls, mock_result_copy, mock_result_copy]

        summary = copy_files_from_dwd_ftp(ftp_host, ftp_path, dst_root)

        # Should be called 3 times: 1 for listing, 2 for copying (alb_rad and t_2m)
        assert mock_run.call_count == 3
        assert isinstance(summary, TransferSummary)
        assert summary.total_transfers == 2
