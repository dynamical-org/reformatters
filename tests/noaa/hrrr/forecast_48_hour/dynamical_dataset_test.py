import pytest

from reformatters.common import validation
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> NoaaHrrrForecast48HourDataset:
    return NoaaHrrrForecast48HourDataset(storage_config=NOOP_STORAGE_CONFIG)


def test_template_config(dataset: NoaaHrrrForecast48HourDataset) -> None:
    """Test that the template config is properly configured."""
    template_config = dataset.template_config

    # Check basic attributes - HRRR uses x/y projected coordinates
    assert template_config.dims == ("init_time", "lead_time", "x", "y")
    assert template_config.append_dim == "init_time"
    assert template_config.append_dim_frequency.total_seconds() == 6 * 3600  # 6 hours

    # Check dataset attributes
    dataset_attrs = template_config.dataset_attributes
    assert dataset_attrs.dataset_id == "noaa-hrrr-forecast-48-hour"
    assert "HRRR" in dataset_attrs.name
    assert "48 hour" in dataset_attrs.name

    # Check that we have data variables defined
    data_vars = template_config.data_vars
    assert len(data_vars) > 0
    assert all(hasattr(var, "internal_attrs") for var in data_vars)
    assert all(hasattr(var.internal_attrs, "hrrr_file_type") for var in data_vars)


def test_region_job_class(dataset: NoaaHrrrForecast48HourDataset) -> None:
    """Test that the region job class is properly configured."""
    region_job_class = dataset.region_job_class

    # Check that it has the required methods
    assert hasattr(region_job_class, "generate_source_file_coords")
    assert hasattr(region_job_class, "download_file")
    assert hasattr(region_job_class, "read_data")
    assert hasattr(region_job_class, "operational_update_jobs")

    # Check source groups method
    assert hasattr(region_job_class, "source_groups")

    # Test that we can call source_groups with empty list
    groups = region_job_class.source_groups([])
    assert len(groups) == 0


def test_operational_kubernetes_resources(
    dataset: NoaaHrrrForecast48HourDataset,
) -> None:
    """Test the Kubernetes resource configuration."""
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image-tag"))

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs

    # Check update job
    assert update_cron_job.name == f"{dataset.dataset_id}-operational-update"
    assert update_cron_job.schedule == "30 0,6,12,18 * * *"  # Every 6 hours at :30
    assert update_cron_job.secret_names == dataset.storage_config.k8s_secret_names
    assert "6" in update_cron_job.cpu  # Should be CPU intensive
    assert (
        "24G" in update_cron_job.memory
    )  # Should have high memory for GRIB processing

    # Check validation job
    assert validation_cron_job.name == f"{dataset.dataset_id}-validation"
    assert validation_cron_job.schedule == "30 1,7,13,19 * * *"  # 1 hour after updates
    assert validation_cron_job.secret_names == dataset.storage_config.k8s_secret_names


def test_validators(dataset: NoaaHrrrForecast48HourDataset) -> None:
    """Test that validators are properly configured."""
    validators = tuple(dataset.validators())
    assert len(validators) == 3
    assert all(isinstance(v, validation.DataValidator) for v in validators)


def test_dataset_properties(dataset: NoaaHrrrForecast48HourDataset) -> None:
    """Test basic dataset properties."""
    assert dataset.dataset_id == "noaa-hrrr-forecast-48-hour"
    assert dataset.dataset_id == dataset.template_config.dataset_attributes.dataset_id
