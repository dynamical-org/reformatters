import pytest

from reformatters.common import validation
from reformatters.ecmwf.aifs_ens.forecast.dynamical_dataset import (
    EcmwfAifsEnsForecastDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfAifsEnsForecastDataset:
    return EcmwfAifsEnsForecastDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


def test_operational_kubernetes_resources(
    dataset: EcmwfAifsEnsForecastDataset,
) -> None:
    cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs
    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert update_cron_job.secret_names == [
        dataset.primary_storage_config.k8s_secret_name
    ]
    assert validation_cron_job.secret_names == [
        dataset.primary_storage_config.k8s_secret_name
    ]


def test_validators(dataset: EcmwfAifsEnsForecastDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
