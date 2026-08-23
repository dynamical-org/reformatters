import pytest

from reformatters.common import validation
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import ECDS_VARIABLES
from reformatters.ecmwf.ifs_ens.forecast_46_day_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day15DegreeDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfIfsEnsForecast46Day15DegreeDataset:
    return EcmwfIfsEnsForecast46Day15DegreeDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


def test_validators_check_masked_variables_are_not_all_nan(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset,
) -> None:
    validators = tuple(dataset.validators())

    assert len(validators) == 3
    assert isinstance(validators[1], validation.CheckRecentNans)
    assert isinstance(validators[2], validation.CheckRecentNans)
    assert validators[2].include_vars == validators[1].exclude_vars
    assert validators[2].max_nan_fraction == 0.9999
    assert validators[2].spatial_sampling == "quarter"


def test_archive_contains_every_dataset_source_variable(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset,
) -> None:
    assert {
        data_var.internal_attrs.ecds_variable
        for data_var in dataset.template_config.data_vars
    } == set(ECDS_VARIABLES)
