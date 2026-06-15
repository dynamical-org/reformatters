from datetime import timedelta
from functools import partial

import pytest

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.ecmwf.ifs_ens.forecast_15_day_spatial.dynamical_dataset import (
    EcmwfIfsEnsForecast15DaySpatialDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_spatial.region_job import (
    _S3_LOCATION_PREFIX,
)


@pytest.fixture
def dataset() -> EcmwfIfsEnsForecast15DaySpatialDataset:
    return EcmwfIfsEnsForecast15DaySpatialDataset(
        primary_storage_config=StorageConfig(
            base_path="s3://test-bucket/path", format=DatasetFormat.ICECHUNK
        ),
    )


def test_virtual_container_matches_ref_locations(
    dataset: EcmwfIfsEnsForecast15DaySpatialDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == _S3_LOCATION_PREFIX


def test_operational_kubernetes_resources(
    dataset: EcmwfIfsEnsForecast15DaySpatialDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.secret_names == dataset.store_factory.k8s_secret_names()
    assert update_cron_job.schedule == "45 7 * * *"
    assert update_cron_job.pod_active_deadline == timedelta(hours=1, minutes=30)
    # Virtual operational updates are strictly single-writer.
    assert update_cron_job.workers_total == 1
    assert update_cron_job.parallelism == 1
    assert update_cron_job.suspend is False
    assert validation_cron_job.suspend is False
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    # Validate after the update fire (45 7) + its 1h30m deadline.
    assert validation_cron_job.schedule == "15 9 * * *"


def test_validators(dataset: EcmwfIfsEnsForecast15DaySpatialDataset) -> None:
    [validator] = dataset.validators()
    assert isinstance(validator, partial)
    assert validator.func is validation.check_forecast_current_data
    assert validator.keywords == {"max_latest_init_time_age": timedelta(hours=12)}
