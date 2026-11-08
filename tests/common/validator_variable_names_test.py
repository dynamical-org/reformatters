import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset


@pytest.mark.parametrize("dataset", DYNAMICAL_DATASETS, ids=lambda d: d.dataset_id)
def test_validators_name_only_template_variables(dataset: DynamicalDataset) -> None:  # type: ignore[type-arg]
    """Every variable a validator names must exist in the dataset template, so
    operational validation cannot crash on a renamed or removed variable."""
    known = [var.path for var in dataset.template_config.data_vars]
    for validator in dataset.validators():
        if isinstance(validator, validation.VariableSelection):
            validator.validate_var_names(known)
        if isinstance(validator, validation.CheckVirtualDecodeHealth):
            assert set(validator.allow_all_nan_vars) <= set(known), validator
