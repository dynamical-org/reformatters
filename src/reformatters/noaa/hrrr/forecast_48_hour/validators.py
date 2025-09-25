import pandas as pd
import xarray as xr

from reformatters.common import validation


def check_data_is_current(ds: xr.Dataset) -> validation.ValidationResult:
    """
    Check that the HRRR forecast data is current within the last 24 hours.

    HRRR provides 48-hour forecasts every 6 hours, so we should have
    recent init_time data within the last day.
    """
    init_time_age_threshold = pd.Timedelta("7h")
    now = pd.Timestamp.now()
    latest_init_time = ds["init_time"].max().item()

    # Convert to pandas timestamp for comparison
    latest_init_time = pd.Timestamp(latest_init_time)

    time_since_latest = now - latest_init_time

    if time_since_latest > init_time_age_threshold:
        return validation.ValidationResult(
            passed=False,
            message=f"Latest init_time is {time_since_latest} old (> {init_time_age_threshold})",
        )

    return validation.ValidationResult(
        passed=True,
        message=f"Data is current: latest init_time is {time_since_latest} hours old",
    )


def check_forecast_completeness(ds: xr.Dataset) -> validation.ValidationResult:
    """
    Check that recent forecasts have the expected lead times based on init_time hour.

    HRRR provides:
    - 18-hour forecasts for most init times
    - 48-hour forecasts for 00, 06, 12, 18 UTC init times
    """
    # Check the last few init_times
    recent_ds = ds.isel(init_time=slice(-4, None))

    problems = []
    for init_time in recent_ds["init_time"].values:
        init_time_pd = pd.Timestamp(init_time)
        init_hour = init_time_pd.hour

        # Expected max lead time based on init hour
        if init_hour in [0, 6, 12, 18]:
            expected_max_lead_time = pd.Timedelta("48h")
        else:
            expected_max_lead_time = pd.Timedelta("18h")

        # Check actual max lead time for this init_time
        init_data = recent_ds.sel(init_time=init_time)
        actual_lead_times = init_data["lead_time"].values

        # Find the maximum lead time that has non-NaN data
        max_lead_with_data = None
        for lead_time in sorted(actual_lead_times, reverse=True):
            lead_data = init_data.sel(lead_time=lead_time)
            if not all(da.isnull().all().item() for da in lead_data.data_vars.values()):
                max_lead_with_data = pd.Timedelta(lead_time)
                break

        if max_lead_with_data is None:
            problems.append(f"No data found for init_time {init_time_pd}")
        elif max_lead_with_data < expected_max_lead_time:
            problems.append(
                f"Init_time {init_time_pd} (hour {init_hour:02d}): "
                f"expected {expected_max_lead_time}, got {max_lead_with_data}"
            )

    if problems:
        return validation.ValidationResult(
            passed=False,
            message="Forecast completeness issues:\n"
            + "\n".join(f"- {p}" for p in problems),
        )

    return validation.ValidationResult(
        passed=True, message="Recent forecasts have expected lead time completeness"
    )


def check_spatial_coverage(
    ds: xr.Dataset,
    max_nan_percent: float = 0.5,  # half of 1%
) -> validation.ValidationResult:
    """
    Check that the data covers the expected HRRR CONUS domain.

    HRRR should have minimal NaN values over CONUS, as it's designed
    for the continental United States.
    """
    # Sample the latest init_time
    sample_ds = ds.isel(init_time=-1)

    problems = []
    for var_name, da in sample_ds.data_vars.items():
        # skip lead_time=0 for accumulations
        if da.attrs["step_type"] != "instant":
            da = da.isel(lead_time=slice(1, None))  # noqa: PLW2901
        nan_percentage = da.isnull().mean().compute().item() * 100

        # HRRR over CONUS should have very few NaN values
        if nan_percentage > max_nan_percent:
            problems.append(f"{var_name}: {nan_percentage:.1f}% NaN values")

    if problems:
        return validation.ValidationResult(
            passed=False,
            message="Excessive NaN values found:\n"
            + "\n".join(f"- {p}" for p in problems),
        )

    return validation.ValidationResult(
        passed=True, message="Perfecnt NaN values are within acceptable limit"
    )
