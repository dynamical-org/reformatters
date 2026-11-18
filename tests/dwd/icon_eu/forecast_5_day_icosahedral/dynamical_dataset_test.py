import itertools
from collections.abc import Sequence
from datetime import timedelta
from pathlib import PurePosixPath
from unittest.mock import Mock, call, patch

import pytest
from typer.testing import CliRunner

from reformatters.dwd.icon_eu.forecast_5_day_icosahedral import (
    DwdIconEuForecast5DayIcosahedralDataset,
)
from reformatters.dwd.icon_eu.forecast_5_day_icosahedral.dynamical_dataset import (
    ARCHIVE_GRIB_FILES_DEADLINE,
    ICOSAHEDRAL_LEVEL_TYPES,
    ICOSAHEDRAL_NWP_INIT_HOURS,
    ICOSAHEDRAL_RUN_ALLOWANCE,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> DwdIconEuForecast5DayIcosahedralDataset:
    return DwdIconEuForecast5DayIcosahedralDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


def test_native_update_uses_native_timing_and_runs(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
) -> None:
    _archive_job, update = dataset.operational_kubernetes_resources("test-image")
    assert update.name == f"{dataset.dataset_id}-update"
    assert update.suspend is False
    assert update.schedule == "17 3,9,15,21 * * *"
    assert update.pod_active_deadline == timedelta(hours=6)
    assert update.active_deadline == timedelta(hours=6)


def test_archive_job_keeps_its_name_and_runs_this_datasets_cli(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
) -> None:
    archive_job, _update = dataset.operational_kubernetes_resources("test-image")

    assert archive_job.name == "dwd-icon-eu-forecast-5-day-archive-grib-files"
    assert archive_job.dataset_id == dataset.dataset_id
    assert archive_job.command == ["archive-grib-files"]
    assert archive_job.suspend is False
    assert archive_job.schedule == "0 4,10,16,22 * * *"
    assert archive_job.pod_active_deadline == ARCHIVE_GRIB_FILES_DEADLINE
    # The next fire is 6h later, and the CronJob's concurrencyPolicy is Replace.
    assert archive_job.pod_active_deadline < timedelta(hours=6)


MODULE = "reformatters.dwd.icon_eu.forecast_5_day_icosahedral.dynamical_dataset"


def _archive_grib_files(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
    phases: Mock | None = None,
    secret: dict[str, str] | None = None,
    monotonic_seconds: Sequence[float] = (0.0,),
) -> Mock:
    """Run `archive_grib_files` with both copy phases mocked, returning the mock
    whose `regular_lat_lon` and `icosahedral` children recorded the calls in order.
    `monotonic_seconds` are successive `time.monotonic()` readings; the last repeats."""
    phases = phases or Mock()
    monotonic = Mock(
        side_effect=itertools.chain(
            monotonic_seconds, itertools.repeat(monotonic_seconds[-1])
        )
    )
    with (
        patch(f"{MODULE}.copy_files_from_dwd_https", phases.regular_lat_lon),
        patch(f"{MODULE}.copy_icosahedral_files_from_dwd_https", phases.icosahedral),
        patch(f"{MODULE}.kubernetes.load_secret", return_value=secret or {}),
        patch(f"{MODULE}.time.monotonic", monotonic),
    ):
        dataset.archive_grib_files(reformat_job_name="test")
    return phases


def test_archive_grib_files_copies_regular_lat_lon_then_icosahedral(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
) -> None:
    phases = _archive_grib_files(dataset)

    assert [name for name, _args, _kwargs in phases.mock_calls] == [
        "regular_lat_lon"
    ] * 4 + ["icosahedral"]
    for i, init_hour in enumerate([0, 6, 12, 18]):
        assert phases.regular_lat_lon.call_args_list[i] == call(
            src_host="https://opendata.dwd.de",
            src_root_path=PurePosixPath(f"/weather/nwp/icon-eu/grib/{init_hour:02d}"),
            dst_root_path=PurePosixPath(dataset.dynamical_grib_archive_rclone_root),
            transfer_parallelism=64,
            checkers=32,
            stats_logging_freq="1m",
            env_vars=None,
        )
    phases.icosahedral.assert_called_once_with(
        dst_root_path=dataset.dynamical_icosahedral_grib_archive_rclone_root,
        nwp_init_hours=ICOSAHEDRAL_NWP_INIT_HOURS,
        level_types=ICOSAHEDRAL_LEVEL_TYPES,
        params=(),
        time_budget=ARCHIVE_GRIB_FILES_DEADLINE - ICOSAHEDRAL_RUN_ALLOWANCE,
        transfer_parallelism=64,
        checkers=32,
        stats_logging_freq="1m",
        env_vars=None,
    )


def test_archive_grib_files_passes_s3_credentials(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
) -> None:
    secret = {"key": "test-key", "secret": "test-secret"}
    phases = _archive_grib_files(dataset, secret=secret)

    assert len(phases.mock_calls) == 5
    for _name, _args, kwargs in phases.mock_calls:
        env_vars = kwargs["env_vars"]
        assert env_vars is not None
        assert env_vars["RCLONE_S3_ACCESS_KEY_ID"] == "test-key"
        assert env_vars["RCLONE_S3_SECRET_ACCESS_KEY"] == "test-secret"  # noqa: S105
        assert env_vars["RCLONE_S3_PROVIDER"] == "AWS"


def test_archive_grib_files_gives_icosahedral_phase_the_time_left_before_the_deadline(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
) -> None:
    regular_lat_lon_seconds = timedelta(hours=3, minutes=31).total_seconds()
    # Readings: job start, regular lat/lon phase start, then its end, which repeats.
    phases = _archive_grib_files(
        dataset, monotonic_seconds=(0.0, 0.0, regular_lat_lon_seconds)
    )

    assert phases.icosahedral.call_args.kwargs["time_budget"] == (
        ARCHIVE_GRIB_FILES_DEADLINE
        - ICOSAHEDRAL_RUN_ALLOWANCE
        - timedelta(hours=3, minutes=31)
    )


def test_archive_grib_files_runs_icosahedral_phase_when_regular_lat_lon_fails(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
) -> None:
    phases = Mock()
    phases.regular_lat_lon.side_effect = RuntimeError("DWD lat/lon listing failed")
    with pytest.raises(RuntimeError, match="DWD lat/lon listing failed"):
        _archive_grib_files(dataset, phases=phases)

    phases.icosahedral.assert_called_once()


def test_archive_grib_files_raises_both_errors_when_both_phases_fail(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
) -> None:
    phases = Mock()
    phases.regular_lat_lon.side_effect = RuntimeError("lat/lon failed")
    phases.icosahedral.side_effect = RuntimeError("icosahedral failed")
    with pytest.raises(ExceptionGroup) as exc_info:
        _archive_grib_files(dataset, phases=phases)

    assert [str(e) for e in exc_info.value.exceptions] == [
        "lat/lon failed",
        "icosahedral failed",
    ]


def test_cli_runs_the_archive_jobs_command(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
) -> None:
    archive_job, _update = dataset.operational_kubernetes_resources("test-image")
    result = CliRunner().invoke(dataset.get_cli(), [*archive_job.command, "--help"])
    assert result.exit_code == 0, result.output


def test_archive_grib_files_fails_when_regular_lat_lon_copyurl_fails(
    dataset: DwdIconEuForecast5DayIcosahedralDataset,
) -> None:
    phases = Mock()
    copier = "reformatters.dwd.archive_gribs.copy_files_from_dwd"
    with (
        patch(
            f"{copier}.list_grib_files_on_dwd_https",
            return_value=[
                PurePosixPath(
                    "t_2m/icon-eu_europe_regular-lat-lon_single-level_2026091500_000_T_2M.grib2.bz2"
                )
            ],
        ),
        patch(
            f"{copier}.list_files_on_dst_for_all_nwp_runs_available_from_dwd",
            return_value=set(),
        ),
        patch(
            "reformatters.common.rclone.run_command_with_concurrent_logging",
            return_value=1,
        ) as run_rclone,
        patch(f"{MODULE}.copy_icosahedral_files_from_dwd_https", phases.icosahedral),
        patch(f"{MODULE}.kubernetes.load_secret", return_value={}),
        patch("reformatters.common.retry.time.sleep"),
        pytest.raises(RuntimeError, match="rclone copyurl exited with code 1"),
    ):
        dataset.archive_grib_files(reformat_job_name="test", nwp_init_hours=[0])

    assert run_rclone.call_count == 3
    phases.icosahedral.assert_called_once()
