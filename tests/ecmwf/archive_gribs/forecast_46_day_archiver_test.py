import pandas as pd
import pytest
from typer.testing import CliRunner

from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import (
    EARLIEST_INIT_TIME,
    ECDS_VARIABLES,
    EcmwfIfsEns46DayGribArchiver,
)
from reformatters.ecmwf.archive_gribs.request_shards import initialization_selections

runner = CliRunner()


def test_operational_kubernetes_resources_is_one_unsuspended_archive_cron() -> None:
    archiver = EcmwfIfsEns46DayGribArchiver()
    (cron_job,) = archiver.operational_kubernetes_resources("test-image")

    assert cron_job.name == "ecmwf-ifs-ens-46-day-gribs-archive-grib-files"
    assert len(cron_job.name) <= 52
    assert cron_job.command == ["archive-grib-files"]
    assert cron_job.dataset_id == archiver.dataset_id
    assert not cron_job.suspend


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


def test_archiver_is_not_a_dataset_and_defines_no_reformat_crons() -> None:
    """The archive has no store, so it must not deploy update or validate crons."""
    cron_jobs = EcmwfIfsEns46DayGribArchiver().operational_kubernetes_resources(
        "test-image"
    )
    assert all(
        not isinstance(c, ReformatCronJob | ValidationCronJob) for c in cron_jobs
    )
    assert all(isinstance(c, CronJob) for c in cron_jobs)
