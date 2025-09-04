import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> NoaaHrrrForecast48HourDataset:
    return NoaaHrrrForecast48HourDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    dataset: NoaaHrrrForecast48HourDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Local backfill reformat
    dataset.backfill_local(append_dim_end=pd.Timestamp("2018-07-13T18:00"))
    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    assert ds.time.max() == pd.Timestamp("2018-07-13T12:00")

    # Operational update
    monkeypatch.setattr(
        dataset.region_job_class,
        "_update_append_dim_end",
        lambda: pd.Timestamp("2018-07-14T00:00"),
    )
    monkeypatch.setattr(
        dataset.region_job_class,
        "_update_append_dim_start",
        lambda existing_ds: pd.Timestamp(existing_ds.time.max().item()),
    )

    dataset.update("test-update")

    # Check resulting dataset
    updated_ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)

    np.testing.assert_array_equal(
        updated_ds.time,
        pd.date_range(
            "2018-07-13T12:00",
            "2018-07-14T00:00",
            freq=dataset.template_config.append_dim_frequency,
        ),
    )
    subset_ds = updated_ds.sel(latitude=50, longitude=-90, method="nearest").sel(
        lead_time=slice("0h", "3h")
    )
    np.testing.assert_array_equal(
        subset_ds["temperature_2m"].values, [190.0, 163.0, 135.0]
    )


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrForecast48HourDataset,
) -> None:
    """Test the Kubernetes resource configuration."""
    # Remove when we re-enable operational resources
    with pytest.raises(NotImplementedError):
        cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))  # noqa: F841

    # assert len(cron_jobs) == 2
    # update_cron_job, validation_cron_job = cron_jobs

    # # Check update job
    # assert update_cron_job.name == f"{dataset.dataset_id}-operational-update"
    # assert update_cron_job.schedule == "30 0,6,12,18 * * *"  # Every 6 hours at :30
    # assert update_cron_job.secret_names == [dataset.storage_config.k8s_secret_name]
    # assert "6" in update_cron_job.cpu
    # assert (
    #     "24G" in update_cron_job.memory
    # )

    # # Check validation job
    # assert validation_cron_job.name == f"{dataset.dataset_id}-validation"
    # assert validation_cron_job.schedule == "30 1,7,13,19 * * *"  # 1 hour after updates
    # assert validation_cron_job.secret_names == [dataset.storage_config.k8s_secret_name]


def test_validators(dataset: NoaaHrrrForecast48HourDataset) -> None:
    """Test that validators are properly configured."""
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    assert all(isinstance(v, validation.DataValidator) for v in validators)
