from datetime import timedelta

import pytest
from typer.testing import CliRunner

from reformatters.common.kubernetes import ReformatCronJob
from reformatters.common.staging import (
    _MAX_KUBERNETES_NAME_LENGTH,
    rename_cronjob_for_staging,
    staging_branch_name,
    staging_cronjob_name,
    staging_cronjob_names,
)


def _make_cronjob(name: str) -> ReformatCronJob:
    return ReformatCronJob(
        name=name,
        schedule="0 * * * *",
        image="test:latest",
        dataset_id="test-dataset",
        cpu="1",
        memory="1G",
        pod_active_deadline=timedelta(minutes=10),
    )


class TestStagingCronjobName:
    def test_basic(self) -> None:
        assert (
            staging_cronjob_name("noaa-gfs-forecast", "0.3.0", "update")
            == "staging-noaa-gfs-forecast-v0-3-0-update"
        )

    def test_dots_replaced_with_dashes(self) -> None:
        name = staging_cronjob_name("test", "1.2.3", "validate")
        assert "." not in name
        assert name == "staging-test-v1-2-3-validate"


class TestRenameCronjobForStaging:
    def test_renames_update_cronjob(self) -> None:
        cronjob = _make_cronjob("noaa-gfs-forecast-update")
        result = rename_cronjob_for_staging(cronjob, "noaa-gfs-forecast", "0.3.0")
        assert result.name == "staging-noaa-gfs-forecast-v0-3-0-update"

    def test_renames_validate_cronjob(self) -> None:
        cronjob = _make_cronjob("noaa-gfs-forecast-validate")
        result = rename_cronjob_for_staging(cronjob, "noaa-gfs-forecast", "0.3.0")
        assert result.name == "staging-noaa-gfs-forecast-v0-3-0-validate"

    def test_preserves_other_fields(self) -> None:
        cronjob = _make_cronjob("test-dataset-update")
        result = rename_cronjob_for_staging(cronjob, "test-dataset", "1.0.0")
        assert result.schedule == cronjob.schedule
        assert result.image == cronjob.image
        assert result.dataset_id == cronjob.dataset_id
        assert result.cpu == cronjob.cpu

    def test_asserts_on_unexpected_name_prefix(self) -> None:
        cronjob = _make_cronjob("wrong-prefix-update")
        with pytest.raises(AssertionError, match="Unexpected cronjob name"):
            rename_cronjob_for_staging(cronjob, "noaa-gfs-forecast", "0.3.0")

    def test_asserts_on_name_too_long(self) -> None:
        long_id = "a" * 50
        cronjob = _make_cronjob(f"{long_id}-update")
        with pytest.raises(AssertionError, match="exceeds 63 char"):
            rename_cronjob_for_staging(cronjob, long_id, "0.3.0")

    def test_longest_real_dataset_id_fits(self) -> None:
        # ecmwf-ifs-ens-forecast-15-day-0-25-degree is currently the longest
        dataset_id = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
        cronjob = _make_cronjob(f"{dataset_id}-update")
        result = rename_cronjob_for_staging(cronjob, dataset_id, "0.3.0")
        assert len(result.name) <= _MAX_KUBERNETES_NAME_LENGTH


class TestStagingCronjobNames:
    def test_returns_update_and_validate(self) -> None:
        names = staging_cronjob_names("noaa-gfs-forecast", "0.3.0")
        assert names == [
            "staging-noaa-gfs-forecast-v0-3-0-update",
            "staging-noaa-gfs-forecast-v0-3-0-validate",
        ]


class TestStagingBranchName:
    def test_basic(self) -> None:
        assert (
            staging_branch_name("noaa-gfs-forecast", "0.3.0")
            == "stage/noaa-gfs-forecast/v0.3.0"
        )


class TestDeployCommandsRegistered:
    def test_deploy_commands_in_cli(self) -> None:
        from reformatters.__main__ import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "deploy " in result.output or "deploy\n" in result.output
        assert "deploy-staging" in result.output
        assert "cleanup-staging" in result.output
