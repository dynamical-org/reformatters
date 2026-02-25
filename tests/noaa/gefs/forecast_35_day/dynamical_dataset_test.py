from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.noaa.gefs.forecast_35_day.dynamical_dataset import (
    GefsForecast35DayDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> GefsForecast35DayDataset:
    return GefsForecast35DayDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


def test_operational_kubernetes_resources(dataset: GefsForecast35DayDataset) -> None:
    """Test the Kubernetes resource configuration for GEFS 35-day forecast dataset."""
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    # Check update job
    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.schedule == "0 7 * * *"  # Daily at 7:00 UTC
    assert update_cron_job.secret_names == dataset.store_factory.k8s_secret_names()
    assert update_cron_job.cpu == "6"
    assert update_cron_job.memory.endswith("G")
    assert update_cron_job.shared_memory == "24G"
    assert update_cron_job.ephemeral_storage == "150G"

    # Check validation job
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert validation_cron_job.schedule == "30 11 * * *"  # Daily at 11:30 UTC
    assert validation_cron_job.secret_names == dataset.store_factory.k8s_secret_names()
    assert validation_cron_job.cpu == "3"
    assert validation_cron_job.memory == "30G"


def test_validators(dataset: GefsForecast35DayDataset) -> None:
    """Test that validators are properly configured."""
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert validation.check_forecast_current_data in validators
    assert validation.check_forecast_recent_nans in validators
    assert all(isinstance(v, validation.DataValidator) for v in validators)


def test_template_config(dataset: GefsForecast35DayDataset) -> None:
    """Test basic template configuration."""
    template_config = dataset.template_config
    assert template_config.dataset_id == "noaa-gefs-forecast-35-day"
    assert template_config.dims == (
        "init_time",
        "ensemble_member",
        "lead_time",
        "latitude",
        "longitude",
    )
    assert template_config.append_dim == "init_time"


def test_dataset_id(dataset: GefsForecast35DayDataset) -> None:
    """Test dataset ID is correctly derived from template config."""
    assert dataset.dataset_id == "noaa-gefs-forecast-35-day"


def test_region_job_integration(dataset: GefsForecast35DayDataset) -> None:
    """Test integration between template config and region job."""
    template_config = dataset.template_config
    region_job_class = dataset.region_job_class

    # Template config and region job should have consistent append_dim
    assert template_config.append_dim == "init_time"

    # Region job should have expected max vars per backfill job
    assert region_job_class.max_vars_per_backfill_job == 3

    # Data variables from template config should be compatible with region job
    data_vars = template_config.data_vars
    assert len(data_vars) > 0

    # Test source grouping with template config variables
    groups = region_job_class.source_groups(data_vars)
    assert len(groups) > 0


def test_cli_integration(dataset: GefsForecast35DayDataset) -> None:
    """Test CLI integration."""
    cli = dataset.get_cli()
    assert cli is not None

    # CLI should have expected commands - typer doesn't have list_commands method,
    # but we can check that the cli exists and is a typer app
    assert hasattr(cli, "callback")  # typer.Typer has this attribute


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch, dataset: GefsForecast35DayDataset
) -> None:
    filter_variable_names = [
        "temperature_2m",  # instantaneous
        "precipitation_surface",  # accumulation we deaccumulate
        "maximum_temperature_2m",  # max over window
    ]
    init_time_start = dataset.template_config.append_dim_start
    init_time_end = init_time_start + timedelta(days=1)

    # Trim lead_time and ensemble_member to speed up test
    orig_get_template = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).sel(
            lead_time=slice("0h", "3h"), ensemble_member=slice(0, 1)
        ),
    )

    dataset.backfill_local(
        append_dim_end=init_time_end, filter_variable_names=filter_variable_names
    )
    ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert ds.init_time.min() == init_time_start
    np.testing.assert_array_equal(
        ds.init_time,
        pd.date_range(
            init_time_start,
            init_time_end,
            freq=dataset.template_config.append_dim_frequency,
            inclusive="left",
        ),
    )
    space_subset_ds = ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))

    # These variables are present at all lead times
    assert (
        (space_subset_ds[["temperature_2m"]].isnull().mean() == 0)
        .all()
        .to_array()
        .all()
    )

    # These variables are not present at hour 0
    assert (
        (
            space_subset_ds[
                [
                    "precipitation_surface",
                    "maximum_temperature_2m",
                ]
            ]
            .sel(lead_time=slice("1h", None))
            .isnull()
            .mean()
            == 0
        )
        .all()
        .to_array()
        .all()
    )
    assert (
        (
            space_subset_ds[
                [
                    "precipitation_surface",
                    "maximum_temperature_2m",
                ]
            ]
            # All null at lead time 0
            .isel(lead_time=0)
            .isnull()
            .mean()
            == 1
        )
        .all()
        .to_array()
        .all()
    )

    point_ds = ds.sel(
        latitude=0,
        longitude=0,
        init_time=init_time_start,
        lead_time="3h",
        ensemble_member=1,
    )

    assert point_ds["temperature_2m"] == 23.25
    assert point_ds["precipitation_surface"] == 1.2040138e-05
    assert point_ds["maximum_temperature_2m"] == 23.875

    # Operational update
    # Advance the init_time_end to add more data
    init_time_end = init_time_end + timedelta(hours=12)
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: init_time_end),
    )

    # Dataset updates always update all variables. For the test we hook into get_jobs to limit vars.
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

    np.testing.assert_array_equal(
        updated_ds.init_time,
        pd.date_range(
            init_time_start,
            init_time_end,
            freq=dataset.template_config.append_dim_frequency,
            inclusive="left",
        ),
    )

    space_subset_ds = updated_ds.sel(latitude=slice(10, 0), longitude=slice(0, 10))

    # These variables are present at all lead times
    assert (
        (space_subset_ds[["temperature_2m"]].isnull().mean() == 0)
        .all()
        .to_array()
        .all()
    )

    # These variables are not present at hour 0
    assert (
        (
            space_subset_ds[
                [
                    "precipitation_surface",
                    "maximum_temperature_2m",
                ]
            ]
            .sel(lead_time=slice("1h", None))
            .isnull()
            .mean()
            == 0
        )
        .all()
        .to_array()
        .all()
    )
    assert (
        (
            space_subset_ds[
                [
                    "precipitation_surface",
                    "maximum_temperature_2m",
                ]
            ]
            # All null at lead time 0
            .isel(lead_time=0)
            .isnull()
            .mean()
            == 1
        )
        .all()
        .to_array()
        .all()
    )

    point_ds = updated_ds.sel(
        latitude=0,
        longitude=0,
        init_time=init_time_start,
        lead_time="3h",
        ensemble_member=1,
    )

    assert point_ds["temperature_2m"] == 23.25
    assert point_ds["precipitation_surface"] == 1.2040138e-05
    assert point_ds["maximum_temperature_2m"] == 23.875
