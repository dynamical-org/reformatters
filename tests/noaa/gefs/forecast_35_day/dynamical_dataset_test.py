import pytest

from reformatters.common import validation
from reformatters.noaa.gefs.forecast_35_day.dynamical_dataset import (
    GefsForecast35DayDataset,
)
from reformatters.noaa.gefs.forecast_35_day.region_job import GefsForecast35DayRegionJob
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> GefsForecast35DayDataset:
    return GefsForecast35DayDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG,
        region_job_class=GefsForecast35DayRegionJob,
    )


def test_operational_kubernetes_resources(dataset: GefsForecast35DayDataset) -> None:
    """Test the Kubernetes resource configuration for GEFS 35-day forecast dataset."""
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    # Check update job
    assert update_cron_job.name == f"{dataset.dataset_id}-operational-update"
    assert update_cron_job.schedule == "0 7 * * *"  # Daily at 7:00 UTC
    assert update_cron_job.secret_names == dataset.store_factory.k8s_secret_names()
    assert update_cron_job.cpu == "6"
    assert update_cron_job.memory == "60G"
    assert update_cron_job.shared_memory == "24G"
    assert update_cron_job.ephemeral_storage == "150G"

    # Check validation job
    assert validation_cron_job.name == f"{dataset.dataset_id}-validation"
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
