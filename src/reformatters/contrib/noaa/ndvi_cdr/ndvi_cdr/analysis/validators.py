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


def check_latest_ndvi_usable_nan_percentage(
    ds: xr.Dataset,
) -> validation.ValidationResult:
    """
    Check that the latest NDVI data has a low nan percentage.
    """
    # We spot checked several times and noticed a consistent ndvi_usable nan percentage
    # of ~93%. For reference, the raw NDVI data we saw had nan percentages ~75%. We expect
    # a large NaN percentage for both variables since oceans and other bodies of water are
    # NaNed out. For `ndvi_usable`, we will generally read this this data using a rolling
    # 16 day max, so individual days with high nan percentages, due to e.g., cloud cover, will
    # be smoothed out.
    threshold = 0.94  # 94% of the data is NaN
    ndvi_usable = ds.isel(time=-1).ndvi_usable

    if (percentage_nan := ndvi_usable.isnull().sum() / ndvi_usable.size) >= threshold:
        return validation.ValidationResult(
            passed=False,
            message=f"Latest NDVI data has high nan percentage: {percentage_nan}",
        )

    return validation.ValidationResult(
        passed=True, message="Latest NDVI data has expected nan percentage"
    )
