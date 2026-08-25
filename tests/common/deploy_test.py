import subprocess
import sys
from importlib.metadata import distribution
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from reformatters.common import monitoring


def test_console_entrypoint_dispatch_installs_sigterm_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = next(
        entrypoint
        for entrypoint in distribution("reformatters").entry_points
        if entrypoint.group == "console_scripts" and entrypoint.name == "main"
    )
    assert entrypoint.value == "reformatters.__main__:app"

    install_sigterm_logger = Mock()
    monkeypatch.setattr(monitoring, "install_sigterm_logger", install_sigterm_logger)
    result = CliRunner().invoke(
        entrypoint.load(), ["noaa-hrrr-analysis-virtual", "dataset-urls"]
    )

    assert result.exit_code == 0, result.exception
    install_sigterm_logger.assert_called_once_with()


def test_direct_file_dispatch_installs_sigterm_logger() -> None:
    code = """
import runpy
import sys
from unittest.mock import Mock

from reformatters.common import monitoring

install_sigterm_logger = Mock()
monitoring.install_sigterm_logger = install_sigterm_logger
sys.argv = [
    "src/reformatters/__main__.py",
    "noaa-hrrr-analysis-virtual",
    "dataset-urls",
]
try:
    runpy.run_path("src/reformatters/__main__.py", run_name="__main__")
except SystemExit as error:
    assert error.code == 0
install_sigterm_logger.assert_called_once_with()
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class TestDeployCommandsRegistered:
    def test_deploy_commands_in_cli(self) -> None:
        from reformatters.__main__ import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "deploy " in result.output or "deploy\n" in result.output
        assert "deploy-staging" in result.output
        assert "cleanup-staging" in result.output
