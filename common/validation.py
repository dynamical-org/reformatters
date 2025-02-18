import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Protocol

import numpy as np
import pandas as pd
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


def check_forecast_current_data(ds: xr.Dataset) -> ValidationResult:
    """Check for data in the most recent day. Fails if no data is found."""
    now = pd.Timestamp.now()
    latest_init_time_ds = ds.sel(init_time=slice(now - timedelta(days=1), None))
    if latest_init_time_ds.sizes["init_time"] == 0:
        return ValidationResult(
            passed=False, message="No data found for the latest day"
        )

    return ValidationResult(
        passed=True,
        message="Data found for the latest day",
    )


def check_forecast_recent_nans(
    ds: xr.Dataset, max_nan_percentage: float = 30
) -> ValidationResult:
    """Check for NaN values in the most recent day of data. Fails if more than 70% of sampled data is NaN."""

    now = pd.Timestamp.now()
    # We want to show that the latest init time has valid data going out up to 10 days (we may not have forecasts
    # past that, depending on the ensemble member and init time). To avoid needing to load a rediculous amount of data
    # we'll choose a random lead_time within that range.
    lead_time_day = np.random.randint(0, 10)  # [0, 10) since we add 1 below
    sample_ds = ds.sel(
        init_time=slice(now - timedelta(days=1), None),
        lead_time=slice(
            pd.Timedelta(days=lead_time_day), pd.Timedelta(days=lead_time_day + 1)
        ),
    )

    problem_vars = []
    for var_name, da in sample_ds.data_vars.items():
        nan_percentage = da.isnull().mean().compute() * 100
        if nan_percentage > max_nan_percentage:
            problem_vars.append((var_name, nan_percentage))

    if problem_vars:
        message = "Excessive NaN values found:\n"
        for var, pct in problem_vars:
            message += f"- {var}: {pct:.1f}% NaN\n"
        return ValidationResult(passed=False, message=message)

    return ValidationResult(
        passed=True,
        message=f"All variables have acceptable NaN percentages (<{max_nan_percentage}%) in sampled locations of latest data",
    )
