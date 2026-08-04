from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from reformatters.common import validation
from reformatters.eccc.hrdps.forecast.dynamical_dataset import EcccHrdpsForecastDataset
from tests.chunk_utils import shrink_chunks_and_shards
from tests.common.dynamical_dataset_test import (
    NOOP_STORAGE_CONFIG,
    assert_configured_validators,
)
from tests.xarray_testing import assert_no_nulls


@pytest.fixture
def dataset() -> EcccHrdpsForecastDataset:
    return _make_dataset()


def _make_dataset() -> EcccHrdpsForecastDataset:
    return EcccHrdpsForecastDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


@pytest.mark.slow
def test_backfill_local_and_operational_update(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset()

    filter_variable_names = [
        "temperature_2m",  # instantaneous
        "precipitation_surface",  # accumulation we deaccumulate, no hour 0 file
        "downward_short_wave_radiation_flux_surface",  # accumulated J/m2 we deaccumulate to W/m2
        "snow_water_equivalent_surface",  # applies scale_factor
    ]

    # Trim to first few lead times to speed up test
    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: shrink_chunks_and_shards(
            orig_get_template(end_time).sel(lead_time=slice("0h", "2h"))
        ),
    )

    init_time_start = pd.Timestamp("2026-07-09T00:00")
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2026-07-09T06:00"),
        filter_variable_names=filter_variable_names,
    )

    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert_array_equal(
        backfill_ds["init_time"], np.array([init_time_start], dtype="datetime64[ns]")
    )

    deaccumulated_vars = [
        "precipitation_surface",
        "downward_short_wave_radiation_flux_surface",
    ]
    space_subset_ds = backfill_ds.isel(y=slice(0, 10), x=slice(0, 10))
    assert_no_nulls(
        space_subset_ds[
            [v for v in filter_variable_names if v not in deaccumulated_vars]
        ]
    )
    # Deaccumulated variables are NaN at hour 0 (no previous step to difference)
    assert_no_nulls(
        space_subset_ds[deaccumulated_vars].sel(lead_time=slice("1h", None))
    )
    assert (
        space_subset_ds[deaccumulated_vars]
        .sel(lead_time="0h")
        .isnull()
        .all()
        .to_array()
        .all()
    )

    # Point in the western Canadian mountains with rain, sunshine, and snowpack
    point_ds = backfill_ds.sel(init_time=init_time_start).isel(y=697, x=306)
    assert_array_equal(
        point_ds["temperature_2m"].values,
        np.array([1.078125, 1.0234375, 0.984375], dtype=np.float32),
    )
    np.testing.assert_allclose(
        point_ds["precipitation_surface"].values,
        np.array([np.nan, 0.00074768, 0.00044918], dtype=np.float32),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        point_ds["downward_short_wave_radiation_flux_surface"].values,
        np.array([np.nan, 95.5, 75.5], dtype=np.float32),
    )
    assert_array_equal(
        point_ds["snow_water_equivalent_surface"].values,
        np.array([0.203125, 0.17578125, 0.14648438], dtype=np.float32),
    )

    # Operational update
    dataset = _make_dataset()
    append_dim_end = pd.Timestamp("2026-07-09T12:00")
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda *args, **kwargs: append_dim_end)
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

    updated_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert_array_equal(
        updated_ds["init_time"],
        pd.DatetimeIndex(["2026-07-09T00:00", "2026-07-09T06:00"]),
    )
    assert_array_equal(
        updated_ds["lead_time"], pd.timedelta_range("0h", "2h", freq="1h")
    )

    space_subset_ds = updated_ds.isel(y=slice(0, 10), x=slice(0, 10))
    assert_no_nulls(
        space_subset_ds[
            [v for v in filter_variable_names if v not in deaccumulated_vars]
        ]
    )
    assert_no_nulls(
        space_subset_ds[deaccumulated_vars].sel(lead_time=slice("1h", None))
    )

    point_ds = updated_ds.sel(init_time="2026-07-09T06:00").isel(y=697, x=306)
    assert_array_equal(
        point_ds["temperature_2m"].values,
        np.array([0.765625, 0.85546875, 0.890625], dtype=np.float32),
    )
    np.testing.assert_allclose(
        point_ds["precipitation_surface"].values,
        np.array([np.nan, 0.00015736, 0.00010252], dtype=np.float32),
        rtol=1e-4,
    )
    assert_array_equal(
        point_ds["snow_water_equivalent_surface"].values,
        np.array([0.20019531, 0.19042969, 0.18164062], dtype=np.float32),
    )

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(dataset: EcccHrdpsForecastDataset) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 3
    archive_job, update_cron_job, validation_cron_job = cron_jobs

    assert archive_job.name == f"{dataset.dataset_id}-archive-grib-files"
    assert archive_job.command == ["archive-grib-files"]
    assert archive_job.suspend is False
    assert "source-coop-storage-options-key" in archive_job.secret_names

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.suspend is True  # until the initial backfill is complete
    assert len(update_cron_job.secret_names) > 0

    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert validation_cron_job.suspend is True
    assert len(validation_cron_job.secret_names) > 0


def test_validators(dataset: EcccHrdpsForecastDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)


def test_archive_grib_files_calls_copy_with_defaults(
    dataset: EcccHrdpsForecastDataset,
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
    dataset: EcccHrdpsForecastDataset,
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


def test_get_cli_has_archive_command(dataset: EcccHrdpsForecastDataset) -> None:
    cli = dataset.get_cli()
    callback_names = [
        getattr(cmd.callback, "__name__", None) for cmd in cli.registered_commands
    ]
    assert "archive_grib_files" in callback_names
