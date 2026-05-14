import pandas as pd
import xarray as xr

from reformatters.common import validation
from reformatters.common.logging import get_logger

log = get_logger(__name__)


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


HRRR_EXPECTED_HOUR_0_NAN_VARS: tuple[str, ...] = (
    "precipitation_surface",
    "categorical_freezing_rain_surface",
    "categorical_ice_pellets_surface",
    "categorical_rain_surface",
    "categorical_snow_surface",
    "percent_frozen_precipitation_surface",
)
