import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.mrms.conus_analysis_hourly.dynamical_dataset import (
    NoaaMrmsConusAnalysisHourlyDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG
from tests.xarray_testing import assert_no_nulls


@pytest.fixture
def dataset() -> NoaaMrmsConusAnalysisHourlyDataset:
    return make_dataset()


def make_dataset() -> NoaaMrmsConusAnalysisHourlyDataset:
    return NoaaMrmsConusAnalysisHourlyDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG,
        replica_storage_configs=[
            StorageConfig(
                base_path="s3://replica-bucket/path", format=DatasetFormat.ICECHUNK
            )
        ],
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset()

    filter_variable_names = [
        "precipitation_surface",
        "categorical_precipitation_type_surface",
    ]

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2024-01-15T02:00"),
        filter_start=pd.Timestamp("2024-01-15T00:00"),
        filter_variable_names=filter_variable_names,
    )

    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert_array_equal(
        backfill_ds["time"],
        pd.date_range("2024-01-15T00:00", "2024-01-15T01:00", freq="1h"),
    )

    # categorical_precipitation_type_surface is instant, no deaccumulation NaN
    assert_no_nulls(backfill_ds["categorical_precipitation_type_surface"])

    # precipitation_surface first timestep is NaN from deaccumulation
    point_ds = backfill_ds.isel(latitude=1750, longitude=3500)
    assert np.isnan(point_ds["precipitation_surface"].values[0])
    assert np.isfinite(point_ds["precipitation_surface"].values[1])

    # Operational update
    dataset = make_dataset()
    append_dim_end = pd.Timestamp("2024-01-15T05:00")
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

    updated_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )

    # Should extend beyond the backfill
    assert updated_ds["time"].max() >= pd.Timestamp("2024-01-15T02:00")


def test_operational_kubernetes_resources(
    dataset: NoaaMrmsConusAnalysisHourlyDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert len(update_cron_job.secret_names) > 0
    assert update_cron_job.suspend is True

    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert len(validation_cron_job.secret_names) > 0
    assert validation_cron_job.suspend is True


def test_validators(dataset: NoaaMrmsConusAnalysisHourlyDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)


@pytest.mark.slow
def test_precipitation_not_null_at_shard_boundary() -> None:
    """
    Test that precipitation_surface is not NaN at the start of the 2nd shard.
    Deaccumulation needs a previous value, but shard boundaries should have valid data
    due to processing region buffering.
    """
    dataset = make_dataset()
    config = dataset.template_config

    precip_var = next(v for v in config.data_vars if v.name == "precipitation_surface")
    time_dim_index = config.dims.index("time")
    assert isinstance(precip_var.encoding.shards, tuple)
    time_shard_size = precip_var.encoding.shards[time_dim_index]

    shard_2_start = (
        config.append_dim_start + time_shard_size * config.append_dim_frequency
    )

    assert time_shard_size == 2160

    dataset.backfill_local(
        append_dim_end=shard_2_start + pd.Timedelta(hours=3),
        filter_start=shard_2_start,
        filter_variable_names=["precipitation_surface"],
    )

    ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )

    shard_2_ds = ds.sel(time=slice(shard_2_start, None))
    expected_times = pd.date_range(shard_2_start, periods=3, freq="1h")
    assert_array_equal(shard_2_ds["time"].values, expected_times)

    # All timesteps at start of shard 2 should have valid precipitation
    precip = shard_2_ds["precipitation_surface"].isel(latitude=1750, longitude=3500)
    assert_no_nulls(precip)
