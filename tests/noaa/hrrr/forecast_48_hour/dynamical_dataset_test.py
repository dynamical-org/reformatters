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
    # Trim to first few hours of lead time dimension to speed up test
    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).sel(
            lead_time=slice("0h", "5h")
        ),
    )

    filter_variable_names = [
        "temperature_2m",  # instantaneous
        "precipitation_surface",  # accumulation we deaccumulate
        "downward_short_wave_radiation_flux_surface",  # average over window, available as analysis and forecast
    ]

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
    np.testing.assert_array_equal(
        backfill_ds["init_time"],
        np.array([init_time_start], dtype="datetime64"),
    )
    space_subset_ds = backfill_ds.isel(x=slice(10, 0), y=slice(0, 10))

    # These variables are present at all lead times
    assert (
        (
            space_subset_ds[
                [v for v in filter_variable_names if v != "precipitation_surface"]
            ]
            .isnull()
            .sum()
            == 0
        )
        .all()
        .to_array()
        .all()
    )
    # These variables are not present at hour 0
    assert (
        (
            space_subset_ds[["precipitation_surface"]]
            .sel(lead_time=slice("1h", None))
            .isnull()
            .sum()
            == 0
        )
        .all()
        .to_array()
        .all()
    )
    point_ds = backfill_ds.sel(
        x=0, y=0, init_time=init_time_start, lead_time="2h", method="nearest"
    )

    assert point_ds["temperature_2m"] == 29.5
    # TODO update
    # assert point_ds["precipitation_surface"] == 0.0
    # assert point_ds["downward_short_wave_radiation_flux_surface"] == 0.0

    # Operational update
    append_dim_end = pd.Timestamp("2018-07-14T00:00")  #
    monkeypatch.setattr(
        dataset.region_job_class,
        "_update_append_dim_end",
        lambda: append_dim_end,
    )

    dataset.update("test-update")

    # Check resulting dataset
    updated_ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )

    expected_init_times = (
        pd.date_range(
            init_time_start,
            append_dim_end,
            freq=dataset.template_config.append_dim_frequency,
            inclusive="left",
        ),
    )
    assert expected_init_times == pd.DatetimeIndex(
        ["2018-07-13T12:00", "2018-07-13T18:00"]
    )
    np.testing.assert_array_equal(updated_ds["init_time"], expected_init_times)
    np.testing.assert_array_equal(
        updated_ds["lead_time"], pd.timedelta_range("0h", "5h", freq="1h")
    )

    # Two init times and two lead times at one point
    point_ds = updated_ds.sel(x=400_000, y=760_000, method="nearest").sel(
        lead_time=slice("0h", "1h")
    )
    np.testing.assert_array_equal(
        point_ds["composite_reflectivity"].values, [[25.5, 30.25], [-10.0, -10.0]]
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
