import gc
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
import xarray as xr

from reformatters.common import validation
from reformatters.common.logging import get_logger

log = get_logger(__name__)


def check_data_is_current(ds: xr.Dataset) -> validation.ValidationResult:
    """
    Check that the HRRR analysis data is current within the last few hours.

    HRRR provides analysis data every hour, so we should have
    recent time data within the last few hours.
    """
    time_age_threshold = pd.Timedelta("4h")
    now = pd.Timestamp.now()
    latest_time = ds["time"].max().item()

    latest_time = pd.Timestamp(latest_time)

    time_since_latest = now - latest_time

    if time_since_latest > time_age_threshold:
        return validation.ValidationResult(
            passed=False,
            message=f"Latest time is {time_since_latest} old (> {time_age_threshold})",
        )

    return validation.ValidationResult(
        passed=True,
        message=f"Data is current: latest time is {time_since_latest} old",
    )


def _check_var_nan_percentage(
    ds: xr.Dataset, var_name: str, max_nan_percent: float, isel: dict[str, int | slice]
) -> str | None:
    da = ds[var_name].isel(isel).load()

    nan_percentage = float(da.isnull().mean().item()) * 100

    del da

    if nan_percentage > max_nan_percent:
        return f"{var_name}: {nan_percentage:.1f}% NaN values"
    return None


def check_analysis_recent_nans(
    ds: xr.Dataset,
    max_nan_percent: float = 0.5,
) -> validation.ValidationResult:
    """Check the fraction of null values in the latest time."""
    var_names = list(ds.data_vars.keys())

    log.info("Loading all values in most recent time to check nan percentage...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(
            partial(
                _check_var_nan_percentage,
                ds,
                max_nan_percent=max_nan_percent,
                isel={"time": -1},
            ),
            var_names,
        )

    problems = [r for r in results if r is not None]

    gc.collect()

    if problems:
        return validation.ValidationResult(
            passed=False,
            message="Excessive NaN values found:\n"
            + "\n".join(f"- {p}" for p in problems),
        )

    return validation.ValidationResult(
        passed=True, message="Percent NaN values are within acceptable limit"
    )
