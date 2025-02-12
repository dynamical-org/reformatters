import logging
from collections.abc import Sequence
from typing import Protocol

import pydantic
import xarray as xr

from common.types import StoreLike

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ValidationResult(pydantic.BaseModel):
    """Result of a validation check."""

    passed: bool
    message: str


class DataValidator(Protocol):
    """Protocol for validation functions."""

    def __call__(self, ds: xr.Dataset) -> ValidationResult: ...


def validate_zarr(store: StoreLike, validators: Sequence[DataValidator]) -> None:
    """
    Validate a zarr dataset by running a series of quality checks.

    Args:
        zarr_path: Path to zarr store
        validators: List of validation functions to run.

    Raises:
        ValueError: If any validation checks fail
    """
    logger.info(f"Validating zarr {store}")

    # Open dataset
    ds = xr.open_zarr(store, chunks=None)

    # Run all validators
    failed_validations = []
    for validator in validators:
        result = validator(ds)
        if not result.passed:
            failed_validations.append(result.message)

    if failed_validations:
        raise ValueError(
            "Zarr validation failed:\n"
            + "\n".join(f"- {msg}" for msg in failed_validations)
        )

    logger.info("Zarr validation passed all checks")
