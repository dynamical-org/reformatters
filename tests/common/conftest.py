from typing import Any

import pytest

from reformatters.common.dynamical_dataset import DynamicalDataset
from tests.dataset_helpers import IMPLEMENTED_DATASETS


@pytest.fixture(
    scope="module",
    params=IMPLEMENTED_DATASETS,
    ids=[d.dataset_id for d in IMPLEMENTED_DATASETS],
)
def dataset(request: pytest.FixtureRequest) -> DynamicalDataset[Any, Any]:
    return request.param
