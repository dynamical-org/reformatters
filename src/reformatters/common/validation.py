import itertools
import json
from collections.abc import Sequence
from datetime import timedelta
from functools import partial
from typing import Literal, Protocol, assert_never, runtime_checkable

import numpy as np
import pandas as pd
import pydantic
import xarray as xr
import zarr

from reformatters.common import iterating
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

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
            log.error(f"Failed validation: {result.message}")
            failed_validations.append(result.message)
        else:
            log.info(f"Passed validation: {result.message}")

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
    lead_time_day = _rng.integers(0, 10)  # [0, 10) since we add 1 below
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


def check_analysis_current_data(
    ds: xr.Dataset, max_expected_delay: timedelta = timedelta(hours=12)
) -> ValidationResult:
    """Check for data in the most recent day. Fails if no data is found."""
    now = pd.Timestamp.now()
    latest_init_time_ds = ds.sel(time=slice(now - max_expected_delay, None))
    if latest_init_time_ds.sizes["time"] == 0:
        return ValidationResult(
            passed=False,
            message=f"No data found within {max_expected_delay} of now",
        )

    return ValidationResult(
        passed=True,
        message=f"Data found within {max_expected_delay} of now",
    )


def check_analysis_recent_nans(  # noqa: PLR0912
    ds: xr.Dataset,
    max_expected_delay: timedelta = timedelta(hours=12),
    max_nan_percentage: float = 5,
    spatial_sampling: Literal["random", "quarter"] = "random",
) -> ValidationResult:
    """Check for NaN values in the most recent day of data. Fails if more than max_nan_percentage of sampled data is NaN."""

    now = pd.Timestamp.now()

    if "latitude" in ds.dims and "longitude" in ds.dims:
        x_dim, y_dim = "longitude", "latitude"
    elif "x" in ds.dims and "y" in ds.dims:
        x_dim, y_dim = "x", "y"
    else:
        raise ValueError("Can't infer spatial dimensions from dataset")

    x_size = ds.sizes[x_dim]
    y_size = ds.sizes[y_dim]
    if spatial_sampling == "random":
        # Use positional indexing to sample a small spatial region
        x_idx = _rng.integers(0, max(1, x_size - 2))
        y_idx = _rng.integers(0, max(1, y_size - 2))

        sample_ds = ds.sel(time=slice(now - max_expected_delay, None)).isel(
            {x_dim: slice(x_idx, x_idx + 2), y_dim: slice(y_idx, y_idx + 2)}
        )

    elif spatial_sampling == "quarter":
        if _rng.integers(0, 2) == 0:
            x_slice = slice(0, x_size // 2)
        else:
            x_slice = slice(x_size // 2, x_size)

        if _rng.integers(0, 2) == 0:
            y_slice = slice(0, y_size // 2)
        else:
            y_slice = slice(y_size // 2, y_size)

        sample_ds = ds.sel(time=slice(now - max_expected_delay, None)).isel(
            {x_dim: x_slice, y_dim: y_slice}
        )
    else:
        assert_never(spatial_sampling)

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
    variables_to_check = _rng.choice(
        list(primary_ds.data_vars.keys()), num_variables_to_check, replace=False
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


def check_for_expected_shards(
    store: zarr.abc.store.Store, ds: xr.Dataset
) -> ValidationResult:
    """Check that the expected shards are present in the store."""
    log.info(f"Checking for expected shards in {store}")

    problem_vars = []
    var_missing_shard_indexes = {}

    for var in map(str, ds.data_vars):  # tell the type checker that var is a str
        ordered_dims = ds[var].dims

        shard_counts_per_dim = [
            len(iterating.dimension_slices(ds, str(dim), "shards"))
            for dim in ordered_dims
        ]
        ranges = [range(shard_count) for shard_count in shard_counts_per_dim]
        expected_shard_indexes = {
            "/".join(map(str, index)) for index in itertools.product(*ranges)
        }

        actual_var_shard_indexes = retry(
            partial(_sync_list_shards, store, var),
            max_attempts=3,
        )

        # During operational updates we trim down the dataset to only include
        # data that was fully processed. This means there may be some extra shards present
        # in the store, but the metadata has been trimmed such that they are not exposed.
        # As such, we don't expect these two sets to necessarily be equal, but we do expect
        # that expected_shard_indexes should be a proper subset of actual_var_shard_indexes.
        missing_shard_indexes = expected_shard_indexes - actual_var_shard_indexes
        if len(missing_shard_indexes) > 0:
            # HRRR categorical variables have enough 0s that some shards are not written
            # We will remove this skip when fill_value is updated to nan / write_empty_chunks is true
            if (
                ds.attrs["dataset_id"] == "noaa-hrrr-forecast-48-hour"
                and "categorical" in var
            ):
                log.info(
                    f"Expecting to find fewer than the maximum shards for categorical hrrr variable ({var}) due to fill value 0 and write empty chunks false"
                )
                continue

            problem_vars.append(var)
            var_missing_shard_indexes[var] = sorted(missing_shard_indexes)

    if len(problem_vars) > 0:
        var_missing_shards_json = json.dumps(var_missing_shard_indexes)
        return ValidationResult(
            passed=False,
            message=f"{problem_vars} are missing expected shards: {var_missing_shards_json[:2000]}",
        )

    return ValidationResult(
        passed=True,
        message="All variables have expected shards",
    )


def _sync_list_shards(store: zarr.abc.store.Store, var: str) -> set[str]:
    return zarr.core.sync.sync(_list_shards(store, var))


async def _list_shards(store: zarr.abc.store.Store, var: str) -> set[str]:
    return {key.split(f"{var}/c/")[-1] async for key in store.list_prefix(f"{var}/c/")}
