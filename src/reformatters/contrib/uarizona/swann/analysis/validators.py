import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common import validation

# For regions outside of CONUS, the values in this dataset are expected
# to be NaNs. We sampled various times across the dataset and determined
# the expected number of NaNs to be ~46.425% of the data.
EXPECTED_NAN_PERCENTAGE = 46.425
MAX_NAN_PERCENTAGE = EXPECTED_NAN_PERCENTAGE + 0.001


def check_data_is_current(ds: xr.Dataset) -> validation.ValidationResult:
    """
    Check that the data is current within the last 48 hours.
    """
    # All times in the dataset are set to start of day, so we need to check
    # that there is `time` that is within 48 hours from start of day.
    now = pd.Timestamp.now().floor("D")
    latest_init_time_ds = ds.sel(time=slice(now - pd.Timedelta(hours=48), None))
    if latest_init_time_ds.sizes["time"] == 0:
        return validation.ValidationResult(
            passed=False, message="No data found for the last 48 hours"
        )

    return validation.ValidationResult(
        passed=True,
        message="Data found for the last 48 hours",
    )


def check_latest_time_nans(ds: xr.Dataset) -> validation.ValidationResult:
    """
    Check that the data does not have more than the expected proportion of NaNs
    for the latest time step.
    """
    sample_ds = ds.isel(time=-1)
    return _check_nans_in_ds("check_latest_time_nans", sample_ds)


def check_random_time_within_last_year_nans(
    ds: xr.Dataset,
) -> validation.ValidationResult:
    """
    Check that the data does not have more than the expected proportion of NaNs
    for a random time in the last year, as we pull a years worth of data for each operational
    update of the dataset.
    """
    random_time_index = np.random.choice(365) + 1
    sample_ds = ds.isel(time=-random_time_index)
    return _check_nans_in_ds("check_random_time_within_last_year_nans", sample_ds)


def _check_nans_in_ds(
    check_name: str, sample_ds: xr.Dataset
) -> validation.ValidationResult:
    problem_vars = []
    for var_name, da in sample_ds.data_vars.items():
        nan_percentage = da.isnull().mean().compute() * 100
        if nan_percentage > MAX_NAN_PERCENTAGE:
            problem_vars.append((var_name, nan_percentage))

    if problem_vars:
        message = f"{check_name}: found excessive NaN values:\n"
        for var, pct in problem_vars:
            message += f"- {var}: {pct:.1f}% NaN\n"
        return validation.ValidationResult(passed=False, message=message)

    return validation.ValidationResult(
        passed=True,
        message=f"{check_name}: all variables have acceptable NaN percentages (<{MAX_NAN_PERCENTAGE}%) in sampled location of latest data",
    )
