from datetime import datetime
from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

import pytest

from reformatters.dwd.copy_files_from_dwd_ftp import (
    _compute_copy_plan,
    _copy_batches,
    copy_files_from_dwd_ftp,
    list_ftp_files,
)


@pytest.fixture
def mock_lsf_output() -> list[str]:
    return [
        "alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2",
        "alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_001_ALB_RAD.grib2.bz2",
        "t_2m/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_T_2M.grib2.bz2",
        "pressure-level/ignore_me.grib2",
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
        assert result[0].name.startswith("icon-eu")
        assert isinstance(result[0], PurePosixPath)
        mock_run.assert_called_once()


def test_compute_copy_plan(mock_lsf_output: list[str]) -> None:
    file_list = [PurePosixPath(p) for p in mock_lsf_output]

    plan = _compute_copy_plan(file_list=file_list)

    # Should have 2 batches keyed by (datetime, variable)
    dt = datetime(2026, 1, 14, 0)
    assert len(plan) == 2
    assert (dt, "alb_rad") in plan
    assert (dt, "t_2m") in plan
    assert len(plan[(dt, "alb_rad")]) == 2
    assert len(plan[(dt, "t_2m")]) == 1


def test_compute_copy_plan_with_limit(mock_lsf_output: list[str]) -> None:
    file_list = [PurePosixPath(p) for p in mock_lsf_output]

    # Limit to 1 file per variable.
    plan = _compute_copy_plan(file_list=file_list, max_files_per_nwp_variable=1)

    dt = datetime(2026, 1, 14, 0)
    assert len(plan) == 2
    assert len(plan[(dt, "alb_rad")]) == 1
    assert len(plan[(dt, "t_2m")]) == 1
    # First alb_rad file should be there
    assert PurePosixPath(mock_lsf_output[0]) in plan[(dt, "alb_rad")]
    # Second alb_rad file should be skipped
    assert PurePosixPath(mock_lsf_output[1]) not in plan[(dt, "alb_rad")]
    # t_2m file should be there
    assert PurePosixPath(mock_lsf_output[2]) in plan[(dt, "t_2m")]


def test_copy_batches() -> None:
    ftp_path = PurePosixPath("/ftp")
    dst_root = PurePosixPath("/dst")
    dt = datetime(2026, 1, 14, 0)
    copy_plan = {
        (dt, "alb_rad"): [PurePosixPath("alb_rad/file1.bz2")],
        (dt, "t_2m"): [PurePosixPath("t_2m/file2.bz2")],
    }

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        _copy_batches("host", ftp_path, dst_root, copy_plan)

        assert mock_run.call_count == 2

        # Check first call
        args, _ = mock_run.call_args_list[0]
        cmd = args[0]
        assert "rclone" in cmd
        assert "copy" in cmd
        assert ":ftp:/ftp" in cmd
        assert "/dst/2026-01-14T00Z" in cmd
        assert any(arg.startswith("--files-from-raw") for arg in cmd)

        # Check second call
        args, _ = mock_run.call_args_list[1]
        cmd = args[0]
        assert "/dst/2026-01-14T00Z" in cmd


def test_copy_files_from_dwd_ftp(mock_lsf_output: list[str]) -> None:
    ftp_host = "opendata.dwd.de"
    ftp_path = PurePosixPath("/weather/nwp/icon-eu/grib/00")
    dst_root = PurePosixPath("/dst")

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "\n".join(mock_lsf_output)
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        copy_files_from_dwd_ftp(ftp_host, ftp_path, dst_root)

        # Should be called 3 times: 1 for listing, 2 for copying (alb_rad and t_2m)
        assert mock_run.call_count == 3
