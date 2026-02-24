import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.gfs.analysis.dynamical_dataset import (
    NoaaGfsAnalysisDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG
from tests.xarray_testing import assert_no_nulls


@pytest.fixture
def dataset() -> NoaaGfsAnalysisDataset:
    return make_dataset()


def make_dataset() -> NoaaGfsAnalysisDataset:
    return NoaaGfsAnalysisDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG,
        replica_storage_configs=[
            StorageConfig(
                base_path="s3://replica-bucket/path", format=DatasetFormat.ICECHUNK
            )
        ],
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = make_dataset()

    filter_variable_names = [
        "temperature_2m",
        "precipitation_surface",
    ]

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2021-05-01T03:00"),
        filter_variable_names=filter_variable_names,
    )

    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    time_start = pd.Timestamp("2021-05-01T00:00")
    assert_array_equal(
        backfill_ds["time"],
        pd.date_range(time_start, "2021-05-01T02:00", freq="1h"),
    )
    space_subset_ds = backfill_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))

    # temperature_2m is instant, should have no nulls
    assert_no_nulls(
        space_subset_ds[
            [v for v in filter_variable_names if v != "precipitation_surface"]
        ]
    )
    # precipitation_surface is null at time 0, but valid after
    assert_no_nulls(space_subset_ds["precipitation_surface"].isel(time=slice(1, None)))

    point_ds = backfill_ds.sel(latitude=0, longitude=0, method="nearest")

    assert_allclose(point_ds["temperature_2m"].values, [28.25, 28.0, 27.875])
    assert_allclose(
        point_ds["precipitation_surface"].values, [np.nan, 0.0, 1.74045574e-05]
    )

    dataset = make_dataset()
    # Set "now" to T06:00. update_template_with_results trims the last hour so we expect 5 hours.
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2021-05-01T06:00")),
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

    # Expected times after trimming the last hour
    expected_times = pd.DatetimeIndex(
        [
            "2021-05-01T00:00",
            "2021-05-01T01:00",
            "2021-05-01T02:00",
            "2021-05-01T03:00",
            "2021-05-01T04:00",
        ]
    )
    assert_array_equal(updated_ds["time"], expected_times)

    space_subset_ds = updated_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))
    # temperature_2m should have no nulls after the update
    assert_no_nulls(space_subset_ds["temperature_2m"].isel(time=slice(1, None)))
    # precipitation_surface has NaN at time 0 by design (no previous value to deaccumulate from)
    assert_no_nulls(space_subset_ds["precipitation_surface"].isel(time=slice(1, None)))

    point_ds = updated_ds.sel(latitude=0, longitude=0, method="nearest")
    assert_allclose(
        point_ds["temperature_2m"].values, [28.25, 28.0, 27.875, 28.0, 28.125]
    )
    # First timestep of precip is NaN as expected
    assert_allclose(
        point_ds["precipitation_surface"].values,
        [np.nan, 0.0, 1.74045574e-05, 1.74045574e-05, 0.0],
    )


def test_operational_kubernetes_resources(
    dataset: NoaaGfsAnalysisDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert len(update_cron_job.secret_names) > 0

    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert len(validation_cron_job.secret_names) > 0


def test_validators(dataset: NoaaGfsAnalysisDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)


@pytest.mark.slow
def test_precipitation_not_null_at_shard_boundary() -> None:
    """
    Test that precipitation_surface is not NaN at the start of the 2nd shard.

    When deaccumulating, the first timestep of each shard processing could incorrectly
    get NaN values because deaccumulation needs a previous value to compute the difference.
    However, at shard boundaries (not the dataset start), we should have valid data
    because we buffer the start of the processing region by one step to enable correct deaccumulation.
    """
    dataset = make_dataset()
    config = dataset.template_config

    # Compute shard_2_start from dataset metadata
    precip_var = next(v for v in config.data_vars if v.name == "precipitation_surface")
    time_dim_index = config.dims.index("time")
    assert isinstance(precip_var.encoding.shards, tuple)
    time_shard_size = precip_var.encoding.shards[time_dim_index]

    shard_2_start = config.append_dim_start + (
        time_shard_size * config.append_dim_frequency
    )

    # Verify our computed value matches expected (1440 hours after start)
    assert time_shard_size == 1440
    assert shard_2_start == pd.Timestamp("2021-06-30T00:00")

    dataset.backfill_local(
        # Get first 3 timesteps of 2nd shard (00:00, 01:00, 02:00)
        append_dim_end=pd.Timestamp("2021-06-30T03:00"),
        filter_start=shard_2_start,
        filter_variable_names=["precipitation_surface"],
    )

    ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )

    # Get only the times at the start of shard 2
    shard_2_ds = ds.sel(time=slice(shard_2_start, None))

    expected_times = pd.DatetimeIndex(
        ["2021-06-30T00:00", "2021-06-30T01:00", "2021-06-30T02:00"]
    )
    assert_array_equal(shard_2_ds["time"].values, expected_times)

    # All 3 timesteps at start of shard 2 should have valid (non-NaN) precipitation.
    # The first timestep of the entire dataset (2021-05-01T00:00) is expected to be NaN,
    # but shard boundaries should NOT have NaN values.
    precip = shard_2_ds["precipitation_surface"].isel(latitude=360, longitude=720)
    assert_no_nulls(precip)
