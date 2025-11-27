import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.hrrr.forecast_48_hour import validators


@pytest.fixture
def hrrr_forecast_dataset() -> xr.Dataset:
    """Create a minimal HRRR forecast dataset for testing."""
    init_times = pd.date_range("2024-01-01", periods=4, freq="6h")
    lead_times = pd.timedelta_range("0h", "48h", freq="1h")
    x = np.arange(100)
    y = np.arange(100)

    ds = xr.Dataset(
        {
            "temperature_2m": (
                ["init_time", "lead_time", "y", "x"],
                np.random.randn(
                    len(init_times), len(lead_times), len(y), len(x)
                ).astype(np.float32),
                {"step_type": "instant"},
            ),
            "precipitation_surface": (
                ["init_time", "lead_time", "y", "x"],
                np.random.randn(
                    len(init_times), len(lead_times), len(y), len(x)
                ).astype(np.float32),
                {"step_type": "accum"},
            ),
        },
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "y": y,
            "x": x,
        },
    )
    return ds


def test_check_forecast_recent_nans_no_nans(hrrr_forecast_dataset: xr.Dataset) -> None:
    """Test that check_forecast_recent_nans passes when there are no NaN values."""
    result = validators.check_forecast_recent_nans(hrrr_forecast_dataset)
    assert result.passed
    assert "within acceptable limit" in result.message


def test_check_forecast_recent_nans_with_excessive_nans(
    hrrr_forecast_dataset: xr.Dataset,
) -> None:
    """Test that check_forecast_recent_nans fails when there are excessive NaN values."""
    # Add excessive NaNs to the latest init_time
    ds = hrrr_forecast_dataset.copy(deep=True)
    ds["temperature_2m"].values[-1, :, :50, :] = np.nan

    result = validators.check_forecast_recent_nans(ds, max_nan_percent=0.5)
    assert not result.passed
    assert "Excessive NaN values found" in result.message
    assert "temperature_2m" in result.message


def test_check_forecast_recent_nans_skips_lead_time_0_for_accumulations(
    hrrr_forecast_dataset: xr.Dataset,
) -> None:
    """Test that lead_time=0 is skipped for accumulation variables."""
    ds = hrrr_forecast_dataset.copy(deep=True)
    # Set lead_time=0 to all NaN for accumulation variable
    ds["precipitation_surface"].values[-1, 0, :, :] = np.nan

    result = validators.check_forecast_recent_nans(ds, max_nan_percent=0.5)
    # Should still pass because lead_time=0 is skipped for accumulations
    assert result.passed
