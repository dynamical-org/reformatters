import pandas as pd
import xarray as xr

from reformatters.common import validation


def check_data_is_current(ds: xr.Dataset) -> validation.ValidationResult:
    """
    Check that the data is current within the last 4 days.

    There seems to be a consistent ~3 day lag for this data's availability.
    We'll check that we have recent data within the last 4 days as a conservative check.
    """
    today_start = pd.Timestamp.now().floor("D")
    latest_init_time_ds = ds.sel(time=slice(today_start - pd.Timedelta(days=4), None))
    if latest_init_time_ds.sizes["time"] == 0:
        return validation.ValidationResult(
            passed=False, message="No data found for the last 4 days"
        )

    return validation.ValidationResult(
        passed=True,
        message="Data found for the last 4 days",
    )
