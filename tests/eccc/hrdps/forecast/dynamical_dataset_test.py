from datetime import timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from reformatters.common import validation
from reformatters.eccc.hrdps.forecast.dynamical_dataset import (
    EcccHrdpsForecastDataset,
)
from tests.chunk_utils import shrink_chunks_and_shards
from tests.common.dynamical_dataset_test import (
    NOOP_STORAGE_CONFIG,
    assert_configured_validators,
)
from tests.xarray_testing import assert_no_nulls

FILTER_VARIABLE_NAMES = [
    "temperature_2m",  # instantaneous, passed through
    "wind_direction_10m",  # rounded finely enough to keep the source's 0.1 degree steps
    "precipitation_surface",  # hourly accumulation converted to a rate
    "downward_short_wave_radiation_flux_surface",  # run total differenced to a rate
    "categorical_precipitation_type_surface",  # flag values, no lead time 0
    "snow_water_equivalent_surface",  # applies scale_factor
]
NO_LEAD_TIME_0_VARIABLE_NAMES = [
    "precipitation_surface",
    "downward_short_wave_radiation_flux_surface",
    "categorical_precipitation_type_surface",
]


@pytest.fixture
def dataset() -> EcccHrdpsForecastDataset:
    return _make_dataset()


def _make_dataset() -> EcccHrdpsForecastDataset:
    return EcccHrdpsForecastDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


def _point_values(ds: xr.Dataset, init_time: str) -> xr.Dataset:
    return ds.sel(init_time=pd.Timestamp(init_time)).isel(y=500, x=1200)


@pytest.mark.slow
def test_backfill_local_and_operational_update(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset()
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
        filter_variable_names=FILTER_VARIABLE_NAMES,
    )

    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert_array_equal(
        backfill_ds["init_time"], np.array([init_time_start], dtype="datetime64")
    )

    space_subset_ds = backfill_ds.isel(y=slice(500, 510), x=slice(1200, 1210))
    assert_no_nulls(
        space_subset_ds[
            [v for v in FILTER_VARIABLE_NAMES if v not in NO_LEAD_TIME_0_VARIABLE_NAMES]
        ]
    )
    assert_no_nulls(
        space_subset_ds[NO_LEAD_TIME_0_VARIABLE_NAMES].sel(lead_time=slice("1h", None))
    )
    assert (
        space_subset_ds[NO_LEAD_TIME_0_VARIABLE_NAMES]
        .sel(lead_time="0h")
        .to_dataarray()
        .isnull()
        .all()
    )

    point_ds = _point_values(backfill_ds, "2026-07-09T00:00")
    assert_array_equal(
        point_ds["temperature_2m"].values,
        np.array([4.09375, 4.0625, 4.09375], dtype=np.float32),
    )
    assert_array_equal(
        point_ds["wind_direction_10m"].values,
        np.array([347.8125, 346.625, 338.875], dtype=np.float32),
    )
    assert_allclose(
        point_ds["precipitation_surface"].values,
        np.array([np.nan, 1.1883676e-06, 2.1383166e-06], dtype=np.float32),
    )
    assert_array_equal(
        point_ds["downward_short_wave_radiation_flux_surface"].values,
        np.array([np.nan, 184.0, 82.5], dtype=np.float32),
    )
    assert_array_equal(
        point_ds["categorical_precipitation_type_surface"].values,
        np.array([np.nan, 1.0, 1.0], dtype=np.float32),
    )
    assert_array_equal(
        point_ds["snow_water_equivalent_surface"].values,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    # A snow covered cell, where dropping the kg m-2 to metres conversion would show
    snowy_ds = backfill_ds.sel(init_time=init_time_start).isel(y=386, x=146)
    assert_array_equal(
        snowy_ds["snow_water_equivalent_surface"].values,
        np.array([0.213867188, 0.206054688, 0.200195312], dtype=np.float32),
    )

    # Operational update
    dataset = _make_dataset()
    append_dim_end = pd.Timestamp("2026-07-09T12:00")
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
            *args, **{**kwargs, "filter_variable_names": FILTER_VARIABLE_NAMES}
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

    space_subset_ds = updated_ds.isel(y=slice(500, 510), x=slice(1200, 1210))
    assert_no_nulls(
        space_subset_ds[
            [v for v in FILTER_VARIABLE_NAMES if v not in NO_LEAD_TIME_0_VARIABLE_NAMES]
        ]
    )
    assert_no_nulls(
        space_subset_ds[NO_LEAD_TIME_0_VARIABLE_NAMES].sel(lead_time=slice("1h", None))
    )

    point_ds = _point_values(updated_ds, "2026-07-09T06:00")
    assert_array_equal(
        point_ds["temperature_2m"].values,
        np.array([3.765625, 5.03125, 5.34375], dtype=np.float32),
    )
    # Local midnight, so no short wave radiation and, here, no precipitation
    assert_array_equal(
        point_ds["downward_short_wave_radiation_flux_surface"].values,
        np.array([np.nan, 0.0, 0.0], dtype=np.float32),
    )
    assert_array_equal(
        point_ds["categorical_precipitation_type_surface"].values,
        np.array([np.nan, 6.0, 6.0], dtype=np.float32),
    )

    assert_configured_validators(dataset)


def test_operational_kubernetes_resources(
    dataset: EcccHrdpsForecastDataset,
) -> None:
    archive_grib_files_job, update_cron_job, validation_cron_job = (
        dataset.operational_kubernetes_resources("test-image-tag")
    )

    assert archive_grib_files_job.name == f"{dataset.dataset_id}-archive-grib-files"
    assert archive_grib_files_job.command == ["archive-grib-files"]
    assert archive_grib_files_job.image == "test-image-tag"
    assert archive_grib_files_job.workers_total == 1
    assert archive_grib_files_job.parallelism == 1
    assert "source-coop-storage-options-key" in archive_grib_files_job.secret_names
    assert archive_grib_files_job.suspend is False

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.schedule == "30 4,10,16,22 * * *"
    # One worker for the newest init time and one for the previous, which the update
    # reprocesses. Each is a whole shard, so neither splits further.
    assert update_cron_job.workers_total == 2
    assert update_cron_job.parallelism == 2
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert validation_cron_job.schedule == "0 5,11,17,23 * * *"
    assert update_cron_job.suspend is False
    assert validation_cron_job.suspend is False

    for cron_job in (archive_grib_files_job, update_cron_job, validation_cron_job):
        assert cron_job.schedule.endswith(" * * *")


def test_validators(dataset: EcccHrdpsForecastDataset) -> None:
    validators = tuple(dataset.validators())
    assert validators == (
        validation.CheckCurrentData(max_delay=timedelta(hours=5)),
        validation.CheckRecentNans(append_dim_window=4),
    )


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


def test_get_cli_has_archive_command(
    dataset: EcccHrdpsForecastDataset,
) -> None:
    cli = dataset.get_cli()
    callback_names = [
        getattr(cmd.callback, "__name__", None) for cmd in cli.registered_commands
    ]
    assert "archive_grib_files" in callback_names
