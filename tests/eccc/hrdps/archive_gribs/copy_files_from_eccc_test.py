from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from reformatters.eccc.hrdps.archive_gribs.copy_files_from_eccc import (
    _copy_one_init_hour,
    copy_files_from_eccc_https,
)


@patch(
    "reformatters.eccc.hrdps.archive_gribs.copy_files_from_eccc.run_command_with_concurrent_logging"
)
def test_copy_one_init_hour_builds_expected_command(mock_run_cmd: MagicMock) -> None:
    mock_run_cmd.return_value = 0

    _copy_one_init_hour(
        src_path="/20260704/WXO-DD/model_hrdps/continental/2.5km/00",
        dst_path=":s3:bucket/eccc-hrdps-grib/20260704/00",
        transfer_parallelism=8,
        checkers=4,
        stats_logging_freq="1m",
        env_vars={},
    )

    mock_run_cmd.assert_called_once()
    cmd = mock_run_cmd.call_args[0][0]
    assert cmd[0] == "/usr/bin/rclone"
    assert cmd[1] == "copy"
    assert cmd[2] == ":http:/20260704/WXO-DD/model_hrdps/continental/2.5km/00"
    assert cmd[3] == ":s3:bucket/eccc-hrdps-grib/20260704/00"
    assert "--http-url=https://dd.weather.gc.ca" in cmd
    assert "--ignore-existing" in cmd
    assert "--filter=+ *.grib2" in cmd
    assert "--transfers=8" in cmd
    assert "--checkers=4" in cmd


@patch(
    "reformatters.eccc.hrdps.archive_gribs.copy_files_from_eccc.run_command_with_concurrent_logging"
)
def test_copy_one_init_hour_tolerates_missing_directory(
    mock_run_cmd: MagicMock,
) -> None:
    mock_run_cmd.return_value = 3  # rclone's "directory not found" exit code

    _copy_one_init_hour(
        src_path="/20260704/WXO-DD/model_hrdps/continental/2.5km/18",
        dst_path=":s3:bucket/eccc-hrdps-grib/20260704/18",
        transfer_parallelism=8,
        checkers=4,
        stats_logging_freq="1m",
        env_vars={},
    )


@patch(
    "reformatters.eccc.hrdps.archive_gribs.copy_files_from_eccc.run_command_with_concurrent_logging"
)
def test_copy_one_init_hour_raises_on_other_errors(mock_run_cmd: MagicMock) -> None:
    mock_run_cmd.return_value = 1

    with pytest.raises(RuntimeError, match="exited with code 1"):
        _copy_one_init_hour(
            src_path="/20260704/WXO-DD/model_hrdps/continental/2.5km/18",
            dst_path=":s3:bucket/eccc-hrdps-grib/20260704/18",
            transfer_parallelism=8,
            checkers=4,
            stats_logging_freq="1m",
            env_vars={},
        )


@patch("reformatters.eccc.hrdps.archive_gribs.copy_files_from_eccc._copy_one_init_hour")
def test_copy_files_from_eccc_https_iterates_days_and_hours(
    mock_copy_one: MagicMock,
) -> None:
    with patch("pandas.Timestamp.now", return_value=pd.Timestamp("2026-07-09T12:00Z")):
        copy_files_from_eccc_https(
            dst_root_path=":s3:bucket/eccc-hrdps-grib",
            nwp_init_hours=[0, 12],
            days_back=1,
            transfer_parallelism=8,
            checkers=4,
            stats_logging_freq="1m",
        )

    # 2 days (today + 1 day back) x 2 init hours = 4 calls
    assert mock_copy_one.call_count == 4
    src_paths = {call.kwargs["src_path"] for call in mock_copy_one.call_args_list}
    dst_paths = {call.kwargs["dst_path"] for call in mock_copy_one.call_args_list}
    assert src_paths == {
        "/20260709/WXO-DD/model_hrdps/continental/2.5km/00",
        "/20260709/WXO-DD/model_hrdps/continental/2.5km/12",
        "/20260708/WXO-DD/model_hrdps/continental/2.5km/00",
        "/20260708/WXO-DD/model_hrdps/continental/2.5km/12",
    }
    assert dst_paths == {
        ":s3:bucket/eccc-hrdps-grib/20260709/00",
        ":s3:bucket/eccc-hrdps-grib/20260709/12",
        ":s3:bucket/eccc-hrdps-grib/20260708/00",
        ":s3:bucket/eccc-hrdps-grib/20260708/12",
    }
