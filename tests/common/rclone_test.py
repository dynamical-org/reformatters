import subprocess
from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

import pytest

from reformatters.common.rclone import _tidy_stats, list_files


def test_tidy_stats_valid() -> None:
    line = "2026/01/31 16:15:41 ERROR :    16.342 MiB / 18.818 MiB, 87%, 0 B/s, ETA -"
    expected = "Transferred so far: 16.342 MiB. Recent throughput: 0 B/s"
    assert _tidy_stats(line) == expected


def test_tidy_stats_invalid() -> None:
    with pytest.raises(ValueError, match="Expected a colon"):
        _tidy_stats("some random log line")

    with pytest.raises(ValueError, match="Expected 4 comma-separated values"):
        _tidy_stats("2026/01/31 16:15:41 ERROR : only, three, values")


@patch("subprocess.run")
def test_list_files_success(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        stdout="file1.txt\nfile2.txt\n", stderr="", returncode=0
    )

    result = list_files(path="/some/path", checkers=4)

    assert result == [PurePosixPath("file1.txt"), PurePosixPath("file2.txt")]
    mock_run.assert_called_once()
    assert "/usr/bin/rclone" in mock_run.call_args[0][0]
    assert "--checkers=4" in mock_run.call_args[0][0]


@patch("subprocess.run")
def test_env_vars_extend_the_environment_rather_than_replace_it(
    mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rclone needs HOME, PATH and any proxy settings alongside the credentials we add."""
    mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
    monkeypatch.setenv("PATH", "/sentinel/bin")

    list_files(path="/some/path", checkers=4, env_vars={"RCLONE_S3_PROVIDER": "AWS"})

    env = mock_run.call_args.kwargs["env"]
    assert env["PATH"] == "/sentinel/bin"
    assert env["RCLONE_S3_PROVIDER"] == "AWS"


@patch("subprocess.run")
def test_list_files_directory_not_found(mock_run: MagicMock) -> None:
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=3, cmd="rclone", stderr="directory not found"
    )

    result = list_files(path="/non/existent", checkers=4)
    assert result == []


@patch("subprocess.run")
def test_list_files_other_error(mock_run: MagicMock) -> None:
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="rclone", stderr="some other error"
    )

    with pytest.raises(subprocess.CalledProcessError):
        list_files(path="/error/path", checkers=4)
