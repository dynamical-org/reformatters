import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.validation import check_analysis_recent_nans


def test_check_analysis_recent_nans_passes() -> None:
    """Test passes when most recent time step has low NaN percentage."""
    times = pd.date_range("2024-01-01", periods=3, freq="6H")

    # Create data where only the first time step has NaNs
    data = np.ones((3, 3, 3))
    data[0, :, :] = np.nan  # First time step all NaN

    ds = xr.Dataset(
        {"temperature": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    # Sample only the most recent time step (which has no NaNs)
    result = check_analysis_recent_nans(
        ds,
        max_nan_percentage=5.0,
        sample_ds_fn=lambda dx: dx.isel(time=slice(-1, None)),
    )

    assert result.passed is True
    assert "acceptable NaN percentages" in result.message


def test_check_analysis_recent_nans_fails() -> None:
    """Test fails when most recent time step has high NaN percentage."""
    times = pd.date_range("2024-01-01", periods=3, freq="6H")

    # Create data where the LAST time step has lots of NaNs
    data = np.ones((3, 3, 3))
    data[-1, :, :] = np.nan  # Last time step all NaN (100% NaNs)

    ds = xr.Dataset(
        {"temperature": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    # Sample the most recent time step (which has 100% NaNs)
    result = check_analysis_recent_nans(
        ds,
        max_nan_percentage=5.0,
        sample_ds_fn=lambda dx: dx.isel(time=slice(-1, None)),
    )

    assert result.passed is False
    assert "Excessive NaN values found" in result.message
    assert "temperature: 100.0% NaN" in result.message
