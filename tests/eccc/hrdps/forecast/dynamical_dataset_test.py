from unittest.mock import Mock, patch

import pytest
import sentry_sdk.crons

from reformatters.common.config import Config
from reformatters.eccc.hrdps.forecast.dynamical_dataset import (
    EcccHrdpsForecastTemporalDynamicalDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcccHrdpsForecastTemporalDynamicalDataset:
    return EcccHrdpsForecastTemporalDynamicalDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


def test_operational_kubernetes_resources(
    dataset: EcccHrdpsForecastTemporalDynamicalDataset,
) -> None:
    (archive_grib_files_job,) = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )
    assert archive_grib_files_job.name == f"{dataset.dataset_id}-archive-grib-files"
    assert archive_grib_files_job.command == ["archive-grib-files"]
    assert archive_grib_files_job.image == "test-image-tag"
    assert archive_grib_files_job.workers_total == 1
    assert archive_grib_files_job.parallelism == 1
    assert "source-coop-storage-options-key" in archive_grib_files_job.secret_names
    # Suspended until a manual run against real Source Coop credentials is verified -
    # see reformatters#711. Un-suspending is a one-line follow-up PR.
    assert archive_grib_files_job.suspend is True


def test_validators(dataset: EcccHrdpsForecastTemporalDynamicalDataset) -> None:
    # TemplateConfig's real coordinates/variables aren't designed yet - see reformatters#711.
    with pytest.raises(NotImplementedError):
        dataset.validators()


def test_archive_grib_files_calls_copy_with_defaults(
    dataset: EcccHrdpsForecastTemporalDynamicalDataset,
) -> None:
    mock_copy = Mock()
    with (
        patch(
            "reformatters.eccc.hrdps.forecast.dynamical_dataset.copy_files_from_eccc_https",
            mock_copy,
        ),
        patch(
            "reformatters.eccc.hrdps.forecast.dynamical_dataset.kubernetes.load_secret",
            return_value={},
        ),
    ):
        dataset.archive_grib_files(reformat_job_name="test")

    mock_copy.assert_called_once_with(
        dst_root_path=":s3:us-west-2.opendata.source.coop/dynamical/eccc-hrdps-grib/",
        nwp_init_hours=(0, 6, 12, 18),
        days_back=1,
        transfer_parallelism=32,
        checkers=16,
        stats_logging_freq="1m",
        env_vars=None,
    )


def test_archive_grib_files_passes_s3_credentials(
    dataset: EcccHrdpsForecastTemporalDynamicalDataset,
) -> None:
    mock_copy = Mock()
    secret = {"key": "test-key", "secret": "test-secret"}
    with (
        patch(
            "reformatters.eccc.hrdps.forecast.dynamical_dataset.copy_files_from_eccc_https",
            mock_copy,
        ),
        patch(
            "reformatters.eccc.hrdps.forecast.dynamical_dataset.kubernetes.load_secret",
            return_value=secret,
        ),
    ):
        dataset.archive_grib_files(reformat_job_name="test")

    env_vars = mock_copy.call_args.kwargs["env_vars"]
    assert env_vars["RCLONE_S3_ACCESS_KEY_ID"] == "test-key"
    assert env_vars["RCLONE_S3_SECRET_ACCESS_KEY"] == "test-secret"  # noqa: S105
    assert env_vars["RCLONE_S3_PROVIDER"] == "AWS"


def test_archive_grib_files_sends_cron_checkins(
    dataset: EcccHrdpsForecastTemporalDynamicalDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(type(Config), "is_sentry_enabled", True)
    mock_capture = Mock()
    monkeypatch.setattr(sentry_sdk.crons, "capture_checkin", mock_capture)

    with (
        patch(
            "reformatters.eccc.hrdps.forecast.dynamical_dataset.copy_files_from_eccc_https",
            Mock(),
        ),
        patch(
            "reformatters.eccc.hrdps.forecast.dynamical_dataset.kubernetes.load_secret",
            return_value={},
        ),
    ):
        dataset.archive_grib_files(reformat_job_name="test")

    statuses = [c.kwargs["status"] for c in mock_capture.call_args_list]
    assert statuses == ["in_progress", "ok"]
    assert (
        mock_capture.call_args_list[0].kwargs["monitor_slug"]
        == f"{dataset.dataset_id}-archive-grib-files"
    )


def test_get_cli_has_archive_command(
    dataset: EcccHrdpsForecastTemporalDynamicalDataset,
) -> None:
    cli = dataset.get_cli()
    callback_names = [
        getattr(cmd.callback, "__name__", None) for cmd in cli.registered_commands
    ]
    assert "archive_grib_files" in callback_names
