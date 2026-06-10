import pytest

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.gefs.forecast_16_day_spatial.dynamical_dataset import (
    GefsForecast16DaySpatialDataset,
)
from reformatters.noaa.gefs.forecast_16_day_spatial.region_job import (
    _S3_LOCATION_PREFIX,
)


@pytest.fixture
def dataset() -> GefsForecast16DaySpatialDataset:
    return GefsForecast16DaySpatialDataset(
        primary_storage_config=StorageConfig(
            base_path="s3://test-bucket/path", format=DatasetFormat.ICECHUNK
        ),
    )


def test_virtual_container_matches_ref_locations(
    dataset: GefsForecast16DaySpatialDataset,
) -> None:
    (container,) = dataset.icechunk_virtual_config.containers
    assert container.url_prefix == _S3_LOCATION_PREFIX


def test_operational_kubernetes_resources(
    dataset: GefsForecast16DaySpatialDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert update_cron_job.secret_names == dataset.store_factory.k8s_secret_names()
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"


def test_validators(dataset: GefsForecast16DaySpatialDataset) -> None:
    validators = tuple(dataset.validators())
    assert validators == (validation.check_forecast_current_data,)
