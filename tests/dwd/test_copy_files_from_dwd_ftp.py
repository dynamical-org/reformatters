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
    ftp_path = PurePosixPath("/weather/nwp/icon-eu/grib/00")

    plan = _compute_copy_plan(ftp_path, file_list)

    # Should have 1 batch keyed by datetime object
    dt = datetime(2026, 1, 14, 0)
    assert len(plan) == 1
    assert dt in plan
    assert len(plan[dt]) == 3
    assert PurePosixPath(mock_lsf_output[0]) in plan[dt]
    assert PurePosixPath(mock_lsf_output[2]) in plan[dt]


def test_compute_copy_plan_with_limit(mock_lsf_output: list[str]) -> None:
    file_list = [PurePosixPath(p) for p in mock_lsf_output]
    ftp_path = PurePosixPath("/weather/nwp/icon-eu/grib/00")

    # Limit to 1 file per variable.
    # mock_lsf_output has 2 files for alb_rad and 1 for t_2m
    plan = _compute_copy_plan(ftp_path, file_list, max_files_per_variable=1)

    dt = datetime(2026, 1, 14, 0)
    assert len(plan[dt]) == 2  # 1 for alb_rad, 1 for t_2m
    # First alb_rad file should be there
    assert PurePosixPath(mock_lsf_output[0]) in plan[dt]
    # Second alb_rad file should be skipped
    assert PurePosixPath(mock_lsf_output[1]) not in plan[dt]
    # t_2m file should be there
    assert PurePosixPath(mock_lsf_output[2]) in plan[dt]


def test_copy_batches() -> None:
    ftp_path = PurePosixPath("/ftp")
    dst_root = PurePosixPath("/dst")
    dt = datetime(2026, 1, 14, 0)
    copy_plan = {
        dt: [PurePosixPath("alb_rad/file1.bz2"), PurePosixPath("t_2m/file2.bz2")]
    }

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        _copy_batches("host", ftp_path, dst_root, copy_plan)

        assert mock_run.call_count == 1
        args, _ = mock_run.call_args
        cmd = args[0]
        assert "rclone" in cmd
        assert "copy" in cmd
        assert ":ftp:/ftp" in cmd
        assert "/dst/2026-01-14T00Z" in cmd
        assert any(arg.startswith("--files-from-raw") for arg in cmd)


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

        # Should be called twice: once for listing, once for copying
        assert mock_run.call_count == 2
