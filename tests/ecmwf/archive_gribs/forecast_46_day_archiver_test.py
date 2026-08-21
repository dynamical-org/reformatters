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

    assert cron_job.name == "ecmwf-ifs-ens-forecast-46-day-gribs-archive-grib-files"
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
    """One initialization is 12 blobs; the archive's layout is what readers index."""
    selections = initialization_selections(ECDS_VARIABLES)
    assert len(selections) == 12
    assert {v for s in selections for v in s.variables} == set(ECDS_VARIABLES)


def test_archiver_is_not_a_dataset_and_defines_no_reformat_crons() -> None:
    """The archive has no store, so it must not deploy update or validate crons."""
    cron_jobs = EcmwfIfsEns46DayGribArchiver().operational_kubernetes_resources(
        "test-image"
    )
    assert all(
        not isinstance(c, ReformatCronJob | ValidationCronJob) for c in cron_jobs
    )
    assert all(isinstance(c, CronJob) for c in cron_jobs)
