from typing import Any

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.dynamical_dataset import DynamicalDataset


def _template_config_is_implemented(dataset: DynamicalDataset[Any, Any]) -> bool:
    """Returns True if a `dataset`'s TemplateConfig is implemented."""
    try:
        dataset.template_config.dimension_coordinates()
    except NotImplementedError:
        return False
    return True


IMPLEMENTED_DATASETS = [
    d for d in DYNAMICAL_DATASETS if _template_config_is_implemented(d)
]
