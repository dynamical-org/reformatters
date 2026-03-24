from pathlib import PurePosixPath
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.dwd.icon_eu.forecast_5_day.dynamical_dataset import (
    DwdIconEuForecast5DayDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG
from tests.xarray_testing import assert_no_nulls


@pytest.fixture
def dataset() -> DwdIconEuForecast5DayDataset:
    return _make_dataset()


def _make_dataset() -> DwdIconEuForecast5DayDataset:
    return DwdIconEuForecast5DayDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG,
        replica_storage_configs=[
            StorageConfig(
                base_path="s3://replica-bucket/path", format=DatasetFormat.ICECHUNK
            )
        ],
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset()
    # Trim to first few lead times to speed up test
    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).sel(
            lead_time=slice("0h", "2h")
        ),
    )

    filter_variable_names = [
        "temperature_2m",  # instantaneous
        "precipitation_surface",  # accumulation we deaccumulate
        "dew_point_temperature_2m",  # instantaneous, one of the new variables
    ]

    # Local backfill reformat
    init_time_start = pd.Timestamp("2026-02-10T00:00")
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2026-02-10T06:00"),
        filter_variable_names=filter_variable_names,
    )

    # Test backfill result
    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert_array_equal(
        backfill_ds["init_time"],
        np.array([init_time_start], dtype="datetime64"),
    )

    # These variables are present at all lead times
    space_subset_ds = backfill_ds.isel(latitude=slice(0, 10), longitude=slice(0, 10))
    assert_no_nulls(
        space_subset_ds[
            [v for v in filter_variable_names if v != "precipitation_surface"]
        ]
    )
    # Precipitation is NaN at hour 0 (deaccumulated from accumulation)
    assert_no_nulls(
        space_subset_ds[["precipitation_surface"]].sel(lead_time=slice("1h", None))
    )

    # Snapshot-style value assertions at a specific point
    point_ds = (
        backfill_ds.sel(init_time=init_time_start)
        .isel(latitude=300, longitude=700)
        .sel(lead_time=slice("0h", "1h"))
    )
    assert_array_equal(
        point_ds["temperature_2m"].values, np.array([-8.3125, -8.375], dtype=np.float32)
    )
    np.testing.assert_allclose(
        point_ds["precipitation_surface"].values,
        np.array([np.nan, 0.0], dtype=np.float32),
    )
    assert_array_equal(
        point_ds["dew_point_temperature_2m"].values,
        np.array([-9.9375, -10.0], dtype=np.float32),
    )

    # Operational update
    dataset = _make_dataset()
    append_dim_end = pd.Timestamp("2026-02-10T12:00")
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: append_dim_end),
    )
    orig_get_jobs = dataset.region_job_class.get_jobs
    monkeypatch.setattr(
        dataset.region_job_class,
        "get_jobs",
        lambda *args, **kwargs: orig_get_jobs(
            *args, **{**kwargs, "filter_variable_names": filter_variable_names}
        ),
    )

    dataset.update("test-update")

    # Check resulting dataset
    updated_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )

    expected_init_times = pd.date_range(
        init_time_start,
        append_dim_end,
        freq=dataset.template_config.append_dim_frequency,
        inclusive="left",
    )
    assert_array_equal(
        expected_init_times,
        pd.DatetimeIndex(["2026-02-10T00:00", "2026-02-10T06:00"]),
    )
    assert_array_equal(updated_ds["init_time"], expected_init_times)
    assert_array_equal(
        updated_ds["lead_time"], pd.timedelta_range("0h", "2h", freq="1h")
    )

    space_subset_ds = updated_ds.isel(latitude=slice(0, 10), longitude=slice(0, 10))
    assert_no_nulls(
        space_subset_ds[
            [v for v in filter_variable_names if v != "precipitation_surface"]
        ]
    )
    assert_no_nulls(
        space_subset_ds[["precipitation_surface"]].sel(lead_time=slice("1h", None))
    )

    # Snapshot-style value assertions at a specific point after update
    update_init_time = pd.Timestamp("2026-02-10T06:00")
    point_ds = (
        updated_ds.sel(init_time=update_init_time)
        .isel(latitude=300, longitude=700)
        .sel(lead_time=slice("0h", "1h"))
    )
    assert_array_equal(
        point_ds["temperature_2m"].values, np.array([-6.125, -5.625], dtype=np.float32)
    )
    np.testing.assert_allclose(
        point_ds["precipitation_surface"].values,
        np.array([np.nan, 2.0302832e-07], dtype=np.float32),
    )
    assert_array_equal(
        point_ds["dew_point_temperature_2m"].values,
        np.array([-7.46875, -6.9375], dtype=np.float32),
    )


def test_operational_kubernetes_resources(
    dataset: DwdIconEuForecast5DayDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 3
    archive_job, update_cron_job, validation_cron_job = cron_jobs

    assert archive_job.name == f"{dataset.dataset_id}-archive-grib-files"
    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"

    # All schedules should run every day
    for cron_job in cron_jobs:
        assert cron_job.schedule.endswith(" * * *")

    assert len(update_cron_job.secret_names) > 0
    assert len(validation_cron_job.secret_names) > 0


def test_validators(dataset: DwdIconEuForecast5DayDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)


def test_archive_grib_files_calls_copy_for_each_init_hour(
    dataset: DwdIconEuForecast5DayDataset,
) -> None:
    mock_copy = Mock()
    with (
        patch(
            "reformatters.dwd.icon_eu.forecast_5_day.dynamical_dataset.copy_files_from_dwd_https",
            mock_copy,
        ),
        patch(
            "reformatters.dwd.icon_eu.forecast_5_day.dynamical_dataset.kubernetes.load_secret",
            return_value={},
        ),
    ):
        dataset.archive_grib_files(reformat_job_name="test")

    assert mock_copy.call_count == 4
    for i, init_hour in enumerate([0, 6, 12, 18]):
        call_kwargs = mock_copy.call_args_list[i]
        assert call_kwargs == call(
            src_host="https://opendata.dwd.de",
            src_root_path=PurePosixPath(f"/weather/nwp/icon-eu/grib/{init_hour:02d}"),
            dst_root_path=PurePosixPath(dataset.dynamical_grib_archive_rclone_root),
            transfer_parallelism=64,
            checkers=32,
            stats_logging_freq="1m",
            env_vars=None,
        )


def test_archive_grib_files_passes_s3_credentials(
    dataset: DwdIconEuForecast5DayDataset,
) -> None:
    mock_copy = Mock()
    secret = {"key": "test-key", "secret": "test-secret"}
    with (
        patch(
            "reformatters.dwd.icon_eu.forecast_5_day.dynamical_dataset.copy_files_from_dwd_https",
            mock_copy,
        ),
        patch(
            "reformatters.dwd.icon_eu.forecast_5_day.dynamical_dataset.kubernetes.load_secret",
            return_value=secret,
        ),
    ):
        dataset.archive_grib_files(reformat_job_name="test")

    # All calls should include S3 credentials
    for c in mock_copy.call_args_list:
        env_vars = c.kwargs["env_vars"]
        assert env_vars is not None
        assert env_vars["RCLONE_S3_ACCESS_KEY_ID"] == "test-key"
        assert env_vars["RCLONE_S3_SECRET_ACCESS_KEY"] == "test-secret"  # noqa: S105
        assert env_vars["RCLONE_S3_PROVIDER"] == "AWS"


def test_get_cli_has_archive_command(
    dataset: DwdIconEuForecast5DayDataset,
) -> None:
    cli = dataset.get_cli()
    callback_names = [
        getattr(cmd.callback, "__name__", None) for cmd in cli.registered_commands
    ]
    assert "archive_grib_files" in callback_names
