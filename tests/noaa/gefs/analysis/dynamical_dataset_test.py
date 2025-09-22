from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.noaa.gefs.analysis.dynamical_dataset import GefsAnalysisDataset
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> GefsAnalysisDataset:
    return GefsAnalysisDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


def test_operational_kubernetes_resources(dataset: GefsAnalysisDataset) -> None:
    """Test the Kubernetes resource configuration for GEFS analysis dataset."""
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    # Check update job
    assert update_cron_job.name == f"{dataset.dataset_id}-operational-update"
    assert update_cron_job.schedule == "0 0,6,12,18 * * *"  # Every 6 hours
    assert update_cron_job.secret_names == dataset.store_factory.k8s_secret_names()
    assert update_cron_job.cpu == "14"
    assert update_cron_job.memory == "30G"
    assert update_cron_job.shared_memory == "12G"
    assert update_cron_job.ephemeral_storage == "35G"

    # Check validation job
    assert validation_cron_job.name == f"{dataset.dataset_id}-validation"
    assert (
        validation_cron_job.schedule == "30 7,10,13,19 * * *"
    )  # 1.5 hours after updates
    assert validation_cron_job.secret_names == dataset.store_factory.k8s_secret_names()
    assert validation_cron_job.cpu == "1.3"
    assert validation_cron_job.memory == "7G"


def test_validators(dataset: GefsAnalysisDataset) -> None:
    """Test that validators are properly configured."""
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert validation.check_analysis_current_data in validators
    assert validation.check_analysis_recent_nans in validators
    assert all(isinstance(v, validation.DataValidator) for v in validators)


def test_template_config(dataset: GefsAnalysisDataset) -> None:
    """Test basic template configuration."""
    template_config = dataset.template_config
    assert template_config.dataset_id == "noaa-gefs-analysis"
    assert template_config.dims == ("time", "latitude", "longitude")
    assert template_config.append_dim == "time"


def test_dataset_id(dataset: GefsAnalysisDataset) -> None:
    """Test dataset ID is correctly derived from template config."""
    assert dataset.dataset_id == "noaa-gefs-analysis"


def test_region_job_integration(dataset: GefsAnalysisDataset) -> None:
    """Test integration between template config and region job."""
    template_config = dataset.template_config
    region_job_class = dataset.region_job_class

    # Template config and region job should have consistent append_dim
    assert template_config.append_dim == "time"

    # Region job should have expected max vars per backfill job
    assert region_job_class.max_vars_per_backfill_job == 1

    # Data variables from template config should be compatible with region job
    data_vars = template_config.data_vars
    assert len(data_vars) > 0

    # Test source grouping with template config variables
    groups = region_job_class.source_groups(data_vars)
    assert len(groups) > 0


def test_cli_integration(dataset: GefsAnalysisDataset) -> None:
    """Test CLI integration."""
    cli = dataset.get_cli()
    assert cli is not None

    # CLI should have expected commands - typer doesn't have list_commands method,
    # but we can check that the cli exists and is a typer app
    assert hasattr(cli, "callback")  # typer.Typer has this attribute


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch, dataset: GefsAnalysisDataset
) -> None:
    filter_variable_names = [
        "temperature_2m",  # instantaneous
        "precipitation_surface",  # accumulation we deaccumulate
        "maximum_temperature_2m",  # max over window
        "downward_short_wave_radiation_flux_surface",  # average over window
    ]
    init_time_start = dataset.template_config.append_dim_start
    init_time_end = init_time_start + timedelta(days=2)
    dataset.backfill_local(
        append_dim_end=init_time_end, filter_variable_names=filter_variable_names
    )
    ds = xr.open_zarr(
        dataset.store_factory.primary_store(), chunks=None, decode_timedelta=True
    )
    assert ds.time.min() == init_time_start
    np.testing.assert_array_equal(
        ds.time,
        pd.date_range(
            init_time_start,
            init_time_end,
            freq=dataset.template_config.append_dim_frequency,
            inclusive="left",
        ),
    )

    # GEFS analysis starts at hour 3
    space_subset_ds = ds.sel(
        latitude=slice(10, 0),
        longitude=slice(0, 10),
        time=slice(init_time_start + timedelta(hours=3), None),
    )

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
                    "downward_short_wave_radiation_flux_surface",
                ]
            ]
            .isnull()
            .mean()
            == 0
        )
        .all()
        .to_array()
        .all()
    )

    point_ds = ds.sel(
        latitude=0,
        longitude=0,
        time=init_time_start + timedelta(hours=3),
    )

    assert point_ds["temperature_2m"] == 26.125
    assert point_ds["precipitation_surface"] == 0.00032806396
    assert point_ds["maximum_temperature_2m"] == 26.0
    assert point_ds["downward_short_wave_radiation_flux_surface"] == 0.0

    # Operational update
    # Advance the init_time_end to the next day to ensure we get the next day's data
    init_time_end = init_time_end + timedelta(days=1)
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

    # We slice off the last hour because most variables (except precip) lack hour 0 values.
    # (see GefsAnalysisRegionJob.update_template_with_results)
    expected_times = pd.date_range(
        init_time_start,
        init_time_end,
        freq=dataset.template_config.append_dim_frequency,
        inclusive="left",
    )[:-1]
    np.testing.assert_array_equal(updated_ds.time, expected_times)

    space_subset_ds = updated_ds.sel(
        latitude=slice(10, 0),
        longitude=slice(0, 10),
        time=slice(init_time_start + timedelta(hours=3), None),
    )

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
                    "downward_short_wave_radiation_flux_surface",
                ]
            ]
            .isnull()
            .mean()
            == 0
        )
        .all()
        .to_array()
        .all()
    )

    point_ds = updated_ds.sel(
        latitude=0,
        longitude=0,
        time=init_time_start + timedelta(hours=3),
    )

    print(
        point_ds["temperature_2m"].values,
        point_ds["precipitation_surface"].values,
        point_ds["maximum_temperature_2m"].values,
        point_ds["downward_short_wave_radiation_flux_surface"].values,
    )

    assert point_ds["temperature_2m"] == 26.125
    assert point_ds["precipitation_surface"] == 0.00032806396
    assert point_ds["maximum_temperature_2m"] == 26.0
    assert point_ds["downward_short_wave_radiation_flux_surface"] == 0.0
