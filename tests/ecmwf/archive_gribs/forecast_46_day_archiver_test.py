from contextlib import nullcontext
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from reformatters.common.config import Config, Env
from reformatters.common.kubernetes import SERVICE_ACCOUNT, CronJob, ReformatCronJob
from reformatters.ecmwf.archive_gribs import forecast_46_day_archiver as archiver_module
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import (
    ARCHIVE_RCLONE_ROOT,
    EARLIEST_INIT_TIME,
    ECDS_VARIABLES,
    MATERIALIZED_PRODUCT_ECDS_VARIABLES,
    EcmwfIfsEns46DayGribArchiver,
)
from reformatters.ecmwf.archive_gribs.request_shards import initialization_selections
from reformatters.ecmwf.ifs_ens.forecast_46_day_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day15DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_6_hourly_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG

runner = CliRunner()


def test_operational_kubernetes_resources_is_one_unsuspended_archive_cron() -> None:
    archiver = EcmwfIfsEns46DayGribArchiver()
    (cron_job,) = archiver.operational_kubernetes_resources("test-image")

    assert cron_job.name == "ecmwf-ifs-ens-46-day-gribs-archive-grib-files"
    assert len(cron_job.name) <= 52
    assert cron_job.command == ["archive-grib-files"]
    assert cron_job.dataset_id == archiver.dataset_id
    assert not cron_job.suspend
    assert cron_job.service_account_name == SERVICE_ACCOUNT


def test_cron_command_matches_a_registered_cli_command() -> None:
    archiver = EcmwfIfsEns46DayGribArchiver()
    command_names = {
        (command.name or command.callback.__name__).replace("_", "-")  # ty: ignore[unresolved-attribute]
        for command in archiver.get_cli().registered_commands
    }
    for cron_job in archiver.operational_kubernetes_resources("test-image"):
        assert cron_job.command[0] in command_names


def test_cli_archive_grib_files_help_works() -> None:
    result = runner.invoke(EcmwfIfsEns46DayGribArchiver().get_cli(), ["--help"])
    assert result.exit_code == 0, result.output
    assert "archive-grib-files" in result.output


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        # The 06 UTC fire selects the initialization published a couple of hours
        # earlier, then walks back.
        (
            "2026-08-20T06:00:00Z",
            ["2026-08-18", "2026-08-17", "2026-08-16"],
        ),
        # Just before publication, the same run is still on the previous day.
        (
            "2026-08-20T04:00:00Z",
            ["2026-08-17", "2026-08-16", "2026-08-15"],
        ),
    ],
)
def test_init_times_to_archive_is_newest_first(now: str, expected: list[str]) -> None:
    init_times = EcmwfIfsEns46DayGribArchiver().init_times_to_archive(
        3, now=pd.Timestamp(now)
    )
    assert [t.strftime("%Y-%m-%d") for t in init_times] == expected


def test_init_times_to_archive_stops_at_the_earliest_initialization() -> None:
    now = EARLIEST_INIT_TIME.tz_localize("UTC") + pd.Timedelta("53h")
    assert EcmwfIfsEns46DayGribArchiver().init_times_to_archive(3, now=now) == [
        EARLIEST_INIT_TIME
    ]


def test_ecds_variables_shard_into_the_archived_selections() -> None:
    """One initialization is these 16 blobs; the archive's layout is what readers index.

    The names are pinned because a reformatter addresses a blob by name: resharding or
    changing ECDS_VARIABLES renames files that every reader already indexes by.
    """
    selections = initialization_selections(ECDS_VARIABLES)
    assert {v for s in selections for v in s.variables} == set(ECDS_VARIABLES)
    assert {selection.file_name for selection in selections} == {
        "pressure-control_forecast-geopotential_height-1526e788.grib2",
        "pressure-control_forecast-specific_humidity-4cb6cf32.grib2",
        "pressure-perturbed_forecast-geopotential_height-6b24c498.grib2",
        "pressure-perturbed_forecast-specific_humidity-4cb6cf32.grib2",
        "pressure-perturbed_forecast-temperature-f62b6585.grib2",
        "pressure-perturbed_forecast-u_component_of_wind-09d82879.grib2",
        "pressure-perturbed_forecast-v_component_of_wind-b41ec9c0.grib2",
        "pressure-perturbed_forecast-vertical_velocity-6a845ed1.grib2",
        "single_level-control_forecast-10_m_u_component_of_wind-86cda6f2.grib2",
        "single_level-control_forecast-2_m_dewpoint_temperature-c21fee3e.grib2",
        "single_level-control_forecast-convective_precipitation-c91f8c5b.grib2",
        "single_level-control_forecast-maximum_2_m_temperature_in_the_last_6_hours-3c5108f7.grib2",
        "single_level-perturbed_forecast-10_m_u_component_of_wind-86cda6f2.grib2",
        "single_level-perturbed_forecast-2_m_dewpoint_temperature-c21fee3e.grib2",
        "single_level-perturbed_forecast-convective_precipitation-c91f8c5b.grib2",
        "single_level-perturbed_forecast-maximum_2_m_temperature_in_the_last_6_hours-3c5108f7.grib2",
    }


def test_archive_variables_are_configured_by_materialized_product() -> None:
    assert MATERIALIZED_PRODUCT_ECDS_VARIABLES["6-hourly"] == (
        "10_m_u_component_of_wind",
        "10_m_v_component_of_wind",
        "maximum_2_m_temperature_in_the_last_6_hours",
        "minimum_2_m_temperature_in_the_last_6_hours",
        "total_precipitation",
    )
    assert len(MATERIALIZED_PRODUCT_ECDS_VARIABLES["daily"]) == 41
    assert set(MATERIALIZED_PRODUCT_ECDS_VARIABLES["6-hourly"]) < set(
        MATERIALIZED_PRODUCT_ECDS_VARIABLES["daily"]
    )
    assert set(ECDS_VARIABLES) == {
        variable
        for variables in MATERIALIZED_PRODUCT_ECDS_VARIABLES.values()
        for variable in variables
    }


def test_archiver_is_not_a_dataset_and_defines_no_reformat_crons() -> None:
    """The archive has no store, so it must not deploy update crons."""
    cron_jobs = EcmwfIfsEns46DayGribArchiver().operational_kubernetes_resources(
        "test-image"
    )
    assert all(not isinstance(c, ReformatCronJob) for c in cron_jobs)
    assert all(isinstance(c, CronJob) for c in cron_jobs)


@pytest.mark.parametrize(
    ("prod", "destination", "readiness", "expected_triggers"),
    [
        (True, ARCHIVE_RCLONE_ROOT, [True, True], 2),
        (True, ARCHIVE_RCLONE_ROOT, [True, False], 2),
        (True, ARCHIVE_RCLONE_ROOT, [False, True], 0),
        (True, ARCHIVE_RCLONE_ROOT, [], 0),
        (False, ARCHIVE_RCLONE_ROOT, [True], 0),
        (True, ":s3:another-bucket/", [True], 0),
    ],
)
@pytest.mark.parametrize("newest_first", [True, False])
def test_archive_triggers_after_completion(
    monkeypatch: pytest.MonkeyPatch,
    prod: bool,
    destination: str,
    readiness: list[bool],
    expected_triggers: int,
    newest_first: bool,
) -> None:
    monkeypatch.setattr(Config, "env", Env.prod if prod else Env.test)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    init_times = list(pd.date_range("2026-08-10", periods=len(readiness)))[::-1]
    if not newest_first:
        init_times.reverse()
        readiness = readiness.copy()
        readiness.reverse()
    with (
        patch.object(
            EcmwfIfsEns46DayGribArchiver, "_monitor", return_value=nullcontext()
        ),
        patch.object(
            EcmwfIfsEns46DayGribArchiver,
            "init_times_to_archive",
            return_value=init_times,
        ),
        patch.object(archiver_module.kubernetes, "load_secret", return_value=None),
        patch.object(
            archiver_module, "archive_initialization", side_effect=readiness * 2
        ) as archive,
        patch.object(archiver_module.kubernetes, "create_job_from_cronjob") as submit,
    ):
        parent = Mock()
        parent.attach_mock(archive, "archive")
        parent.attach_mock(submit, "submit")
        archiver = EcmwfIfsEns46DayGribArchiver()
        archiver.archive_grib_files("archive-job-123", dst_root_path=destination)
        assert [call[0] for call in parent.mock_calls] == ["archive"] * len(
            readiness
        ) + ["submit"] * expected_triggers
        first_calls = list(submit.call_args_list)
        assert submit.call_count == expected_triggers
        if expected_triggers:
            assert {call.args[0] for call in first_calls} == {
                "ecmwf-ifs-ens-46-day-daily-update",
                "ecmwf-ifs-ens-46-day-6-hourly-update",
            }
            for call in first_calls:
                assert len(call.args[1]) in (39, 42)
                assert len(f"{call.args[1]}-999-xxxxx") <= 63
                assert not call.kwargs
        submit.reset_mock()
        archiver.archive_grib_files("archive-job-123", dst_root_path=destination)
        assert submit.call_args_list == first_calls


def test_trigger_targets_keep_their_cron_backstops() -> None:
    for dataset, name, schedule in (
        (
            EcmwfIfsEnsForecast46Day15DegreeDataset(
                primary_storage_config=NOOP_STORAGE_CONFIG
            ),
            "ecmwf-ifs-ens-46-day-daily-update",
            "0 9 * * *",
        ),
        (
            EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset(
                primary_storage_config=NOOP_STORAGE_CONFIG
            ),
            "ecmwf-ifs-ens-46-day-6-hourly-update",
            "0 10 * * *",
        ),
    ):
        (cron,) = dataset.operational_kubernetes_resources("image")
        assert cron.name == name
        assert cron.schedule == schedule
        assert not cron.suspend
        assert cron.service_account_name == SERVICE_ACCOUNT


def test_archive_failure_does_not_submit_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Config, "env", Env.prod)
    with (
        patch.object(
            EcmwfIfsEns46DayGribArchiver, "_monitor", return_value=nullcontext()
        ),
        patch.object(archiver_module.kubernetes, "load_secret", return_value=None),
        patch.object(
            archiver_module,
            "archive_initialization",
            side_effect=[True, RuntimeError("archive failed")],
        ),
        patch.object(archiver_module.kubernetes, "create_job_from_cronjob") as submit,
    ):
        with pytest.raises(RuntimeError, match="archive failed"):
            EcmwfIfsEns46DayGribArchiver().archive_grib_files("archive-job-123")
        submit.assert_not_called()
