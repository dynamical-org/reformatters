from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Protocol

import numpy as np
import pandas as pd
import pydantic
import xarray as xr
import zarr

from reformatters.common.logging import get_logger

logger = get_logger(__name__)


class ValidationResult(pydantic.BaseModel):
    """Result of a validation check."""

    passed: bool
    message: str


class DataValidator(Protocol):
    """Protocol for validation functions."""

    def __call__(self, ds: xr.Dataset) -> ValidationResult: ...


def validate_zarr(
    store: zarr.storage.StoreLike, validators: Sequence[DataValidator]
) -> None:
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
    """Check for NaN values in the most recent day of data. Fails if more than max_nan_percentage of sampled data is NaN."""

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


def check_analysis_current_data(ds: xr.Dataset) -> ValidationResult:
    """Check for data in the most recent day. Fails if no data is found."""
    now = pd.Timestamp.now()
    latest_init_time_ds = ds.sel(time=slice(now - timedelta(hours=12), None))
    if latest_init_time_ds.sizes["time"] == 0:
        return ValidationResult(
            passed=False, message="No data found for the latest day"
        )

    return ValidationResult(
        passed=True,
        message="Data found for the latest day",
    )


def default_sample_ds_fn(ds: xr.Dataset) -> xr.Dataset:
    """Sample a random location from the dataset."""
    now = pd.Timestamp.now()
    lon, lat = np.random.uniform(-180, 179), np.random.uniform(-90, 89)
    return ds.sel(
        time=slice(now - timedelta(hours=12), None),
        latitude=slice(lat, lat - 2),
        longitude=slice(lon, lon + 2),
    )


def check_analysis_recent_nans(
    ds: xr.Dataset,
    max_nan_percentage: float = 5,
    sample_ds_fn: Callable[[xr.Dataset], xr.Dataset] = default_sample_ds_fn,
) -> ValidationResult:
    """Check for NaN values in the most recent day of data. Fails if more than max_nan_percentage of sampled data is NaN.

    Parameters
    ----------
    ds: xr.Dataset
        The dataset to check.
    max_nan_percentage: float
        The maximum percentage of NaN values allowed in the sampled data. Default is 5%.
    sample_ds_fn: Callable[[xr.Dataset], xr.Dataset]
        A function that takes a dataset and returns a sample dataset to use for validation.
        Default is a dataset within the last 12 hours at a random point in the dataset.
    """

    sample_ds = sample_ds_fn(ds)

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
        message=f"All variables have acceptable NaN percentages (<{max_nan_percentage}%) in sampled location of latest data",
    )
