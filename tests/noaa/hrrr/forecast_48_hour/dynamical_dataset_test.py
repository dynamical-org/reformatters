import pytest

from reformatters.common import validation
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> NoaaHrrrForecast48HourDataset:
    return NoaaHrrrForecast48HourDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


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
