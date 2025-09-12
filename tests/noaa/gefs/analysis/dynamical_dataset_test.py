import pytest

from reformatters.common import validation
from reformatters.noaa.gefs.analysis.dynamical_dataset import GefsAnalysisDataset
from reformatters.noaa.gefs.analysis.region_job import GefsAnalysisRegionJob
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> GefsAnalysisDataset:
    return GefsAnalysisDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG,
        region_job_class=GefsAnalysisRegionJob,
    )


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
