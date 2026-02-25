from typer.testing import CliRunner


class TestDeployCommandsRegistered:
    def test_deploy_commands_in_cli(self) -> None:
        from reformatters.__main__ import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "deploy " in result.output or "deploy\n" in result.output
        assert "deploy-staging" in result.output
        assert "cleanup-staging" in result.output
