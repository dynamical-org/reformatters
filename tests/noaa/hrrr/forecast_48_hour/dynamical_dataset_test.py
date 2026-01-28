import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG
from tests.xarray_testing import assert_no_nulls


@pytest.fixture
def dataset() -> NoaaHrrrForecast48HourDataset:
    return make_dataset()


def make_dataset() -> NoaaHrrrForecast48HourDataset:
    return NoaaHrrrForecast48HourDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG,
        replica_storage_configs=[
            StorageConfig(
                base_path="s3://replica-bucket/path", format=DatasetFormat.ICECHUNK
            )
        ],
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create our first dataset, we'll use a different instance for the backfill and the update
    dataset = make_dataset()
    # Trim to first few hours of lead time dimension to speed up test
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
        "downward_short_wave_radiation_flux_surface",  # average over window, available as analysis and forecast
    ]
    # Uncomment to test all variables
    # filter_variable_names = [var.name for var in dataset.template_config.data_vars]

    # Local backfill reformat
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2018-07-13T18:00"),
        filter_variable_names=filter_variable_names,
    )

    # Test backfill result
    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    init_time_start = pd.Timestamp("2018-07-13T12:00")
    assert_array_equal(
        backfill_ds["init_time"],
        np.array([init_time_start], dtype="datetime64"),
    )
    space_subset_ds = backfill_ds.isel(x=slice(10, 0), y=slice(0, 10))

    # These variables are present at all lead times
    assert_no_nulls(
        space_subset_ds[
            [v for v in filter_variable_names if v != "precipitation_surface"]
        ]
    )
    # These variables are not present at hour 0
    assert_no_nulls(
        space_subset_ds[["precipitation_surface"]].sel(lead_time=slice("1h", None))
    )
    # A point that has non zero precipitation
    point_ds = backfill_ds.sel(init_time=init_time_start, lead_time="2h").isel(
        x=1, y=-2
    )

    assert point_ds["temperature_2m"] == 22.875
    # deaccumulated to rate
    assert point_ds["precipitation_surface"] == 4.720688e-05
    assert point_ds["downward_short_wave_radiation_flux_surface"] == 8.1875

    # Operational update
    dataset = make_dataset()  # fresh one to simulate new process
    append_dim_end = pd.Timestamp("2018-07-14T00:00")
    monkeypatch.setattr(
        dataset.region_job_class,
        "_update_append_dim_end",
        lambda: append_dim_end,
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
        expected_init_times, pd.DatetimeIndex(["2018-07-13T12:00", "2018-07-13T18:00"])
    )
    assert_array_equal(updated_ds["init_time"], expected_init_times)
    assert_array_equal(
        updated_ds["lead_time"], pd.timedelta_range("0h", "2h", freq="1h")
    )

    space_subset_ds = updated_ds.isel(x=slice(-10, 0), y=slice(0, 10))
    # These variables are present at all lead times
    assert_no_nulls(
        space_subset_ds[
            [v for v in filter_variable_names if v != "precipitation_surface"]
        ]
    )
    # These variables are not present at hour 0
    assert_no_nulls(
        space_subset_ds[["precipitation_surface"]].sel(lead_time=slice("1h", None))
    )

    # Two init times and two lead times at one point
    point_ds = updated_ds.sel(x=400_000, y=760_000, method="nearest").sel(
        lead_time=slice("0h", "1h")
    )
    assert_array_equal(
        point_ds["temperature_2m"].values, [[21.0, 21.375], [20.75, 23.75]]
    )
    # deaccumulated to rate
    np.testing.assert_allclose(
        point_ds["precipitation_surface"].values,
        [[np.nan, 0.00018024445], [np.nan, 0.0]],
    )
    assert_array_equal(
        point_ds["downward_short_wave_radiation_flux_surface"].values,
        [[1.296875, 36.75], [912.0, 868.0]],
    )


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrForecast48HourDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert len(update_cron_job.secret_names) > 0

    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert len(validation_cron_job.secret_names) > 0


def test_validators(dataset: NoaaHrrrForecast48HourDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    assert all(isinstance(v, validation.DataValidator) for v in validators)
