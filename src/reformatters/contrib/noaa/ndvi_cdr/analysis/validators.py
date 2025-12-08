import pandas as pd
import xarray as xr

from reformatters.common import validation


def check_data_is_current(ds: xr.Dataset) -> validation.ValidationResult:
    """
    Check that the data is current within the last 14 days.

    There's usually a roughly ~3 day lag for this data's availability.
    Sometimes this goes much higher and it is not an operationally supported dynamical dataset (it's contrib)
    so we'll alert after a much longer lag of days.
    """
    today_start = pd.Timestamp.now().floor("D")
    latest_init_time_ds = ds.sel(time=slice(today_start - pd.Timedelta(days=30), None))
    if latest_init_time_ds.sizes["time"] == 0:
        return validation.ValidationResult(
            passed=False, message="No data found for the allowed delay window"
        )

    return validation.ValidationResult(
        passed=True,
        message="Data found for the allowed delay window",
    )


def check_latest_ndvi_usable_nan_percentage(
    ds: xr.Dataset,
) -> validation.ValidationResult:
    """
    Check that the latest NDVI data has a low nan percentage.
    """
    # We spot checked several times and noticed a consistent ndvi_usable nan percentage
    # of ~93%. A large NaN percentage for both variables is expected because oceans and
    # other bodies of water are always NaN. In practice we've seen at least one day > 96% NaN.
    threshold = 0.97
    ndvi_usable = ds.isel(time=-1).ndvi_usable

    if (percentage_nan := ndvi_usable.isnull().sum() / ndvi_usable.size) >= threshold:
        return validation.ValidationResult(
            passed=False,
            message=f"Latest NDVI data has high nan percentage: {percentage_nan}",
        )

    return validation.ValidationResult(
        passed=True, message="Latest NDVI data has expected nan percentage"
    )
