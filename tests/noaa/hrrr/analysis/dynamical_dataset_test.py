import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.hrrr.analysis.dynamical_dataset import (
    NoaaHrrrAnalysisDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG
from tests.xarray_testing import assert_no_nulls


@pytest.fixture
def dataset() -> NoaaHrrrAnalysisDataset:
    return make_dataset()


def make_dataset() -> NoaaHrrrAnalysisDataset:
    return NoaaHrrrAnalysisDataset(
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
        "downward_short_wave_radiation_flux_surface",
    ]

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2018-07-13T14:00"),
        filter_variable_names=filter_variable_names,
    )

    backfill_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    time_start = pd.Timestamp("2018-07-13T12:00")
    assert_array_equal(
        backfill_ds["time"],
        pd.date_range(time_start, "2018-07-13T13:00", freq="1h"),
    )
    space_subset_ds = backfill_ds.isel(x=slice(-10, 0), y=slice(0, 10))

    assert_no_nulls(
        space_subset_ds[
            [v for v in filter_variable_names if v != "precipitation_surface"]
        ]
    )

    # Check precipitation values - first value should be nan, rest should not be null
    precip_data = backfill_ds["precipitation_surface"]
    assert np.isnan(precip_data.isel(time=0, x=1, y=-2).values)
    # Check that subsequent time steps have non-null values in at least some locations
    assert_no_nulls(
        precip_data.isel(time=slice(1, None)).isel(x=slice(-10, 0), y=slice(0, 10))
    )

    point_ds = backfill_ds.sel(time=time_start).isel(x=1, y=-2)

    assert point_ds["temperature_2m"] == 22.875
    assert point_ds["downward_short_wave_radiation_flux_surface"] == 8.1875

    dataset = make_dataset()
    append_dim_end = pd.Timestamp("2018-07-13T15:00")
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

    updated_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )

    expected_times = pd.date_range(
        time_start,
        append_dim_end,
        freq=dataset.template_config.append_dim_frequency,
        inclusive="left",
    )
    assert_array_equal(
        expected_times,
        pd.DatetimeIndex(["2018-07-13T12:00", "2018-07-13T13:00", "2018-07-13T14:00"]),
    )
    assert_array_equal(updated_ds["time"], expected_times)

    space_subset_ds = updated_ds.isel(x=slice(-10, 0), y=slice(0, 10))
    assert_no_nulls(
        space_subset_ds[
            [v for v in filter_variable_names if v != "precipitation_surface"]
        ]
    )

    point_ds = updated_ds.sel(x=400_000, y=760_000, method="nearest").sel(
        time=slice("2018-07-13T12:00", "2018-07-13T13:00")
    )
    assert_array_equal(point_ds["temperature_2m"].values, [21.0, 21.375])
    assert_array_equal(
        point_ds["downward_short_wave_radiation_flux_surface"].values,
        [1.296875, 36.75],
    )


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrAnalysisDataset,
) -> None:
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert len(update_cron_job.secret_names) > 0

    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert len(validation_cron_job.secret_names) > 0


def test_validators(dataset: NoaaHrrrAnalysisDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
