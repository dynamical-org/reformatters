from collections.abc import Sequence
from datetime import timedelta
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pydantic
import xarray as xr
import zarr

from reformatters.common import iterating
from reformatters.common.logging import get_logger

log = get_logger(__name__)


class ValidationResult(pydantic.BaseModel):
    """Result of a validation check."""

    passed: bool
    message: str


@runtime_checkable
class DataValidator(Protocol):
    """Protocol for validation functions."""

    def __call__(self, ds: xr.Dataset) -> ValidationResult: ...


def validate_dataset(
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
    log.info(f"Validating zarr {store}")

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

    log.info("Zarr validation passed all checks")


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


def check_analysis_recent_nans(
    ds: xr.Dataset, max_nan_percentage: float = 5
) -> ValidationResult:
    """Check for NaN values in the most recent day of data. Fails if more than max_nan_percentage of sampled data is NaN."""

    now = pd.Timestamp.now()

    lon, lat = np.random.uniform(-180, 179), np.random.uniform(-90, 89)
    sample_ds = ds.sel(
        time=slice(now - timedelta(hours=12), None),
        latitude=slice(lat, lat - 2),
        longitude=slice(lon, lon + 2),
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
        message=f"All variables have acceptable NaN percentages (<{max_nan_percentage}%) in sampled location of latest data",
    )


def compare_replica_and_primary(
    append_dim: str, replica_ds: xr.Dataset, primary_ds: xr.Dataset
) -> ValidationResult:
    """Compare the data in the replica and primary stores."""
    problem_coords = []
    for coord in primary_ds.coords:
        try:
            xr.testing.assert_equal(primary_ds[coord], replica_ds[coord])
        except AssertionError as e:
            log.exception(e)
            problem_coords.append(coord)
    if problem_coords:
        message = f"Data in replica and primary stores are different for coords: {problem_coords}"
        return ValidationResult(passed=False, message=message)

    num_variables_to_check = min(5, len(primary_ds.data_vars))
    variables_to_check = np.random.choice(
        list(primary_ds.data_vars.keys()), num_variables_to_check
    )

    last_chunk = iterating.dimension_slices(primary_ds, append_dim, "chunks")[-1]
    replica_ds_last_chunk = replica_ds[variables_to_check].isel(
        {append_dim: last_chunk}
    )
    primary_ds_last_chunk = primary_ds[variables_to_check].isel(
        {append_dim: last_chunk}
    )

    try:
        xr.testing.assert_equal(replica_ds_last_chunk, primary_ds_last_chunk)
    except AssertionError as e:
        log.exception(e)
        return ValidationResult(
            passed=False,
            message=f"Data in replica and primary stores are different for chunks: {last_chunk}",
        )

    return ValidationResult(
        passed=True,
        message="Data in tested subset of replica and primary stores is the same",
    )
