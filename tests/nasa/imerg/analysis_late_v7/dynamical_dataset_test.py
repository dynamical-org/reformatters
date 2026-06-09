import pytest

from reformatters.common import validation
from reformatters.nasa.imerg.analysis_late_v7.dynamical_dataset import (
    NasaImergAnalysisLateV7Dataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> NasaImergAnalysisLateV7Dataset:
    return NasaImergAnalysisLateV7Dataset(primary_storage_config=NOOP_STORAGE_CONFIG)


def test_operational_kubernetes_resources(
    dataset: NasaImergAnalysisLateV7Dataset,
) -> None:
    cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs
    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert "nasa-earthdata" in update_cron_job.secret_names
    assert update_cron_job.shared_memory is not None


def test_validators(dataset: NasaImergAnalysisLateV7Dataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
