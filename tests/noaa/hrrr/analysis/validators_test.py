import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.hrrr.analysis import validators


@pytest.fixture
def hrrr_analysis_dataset(rng: np.random.Generator) -> xr.Dataset:
    """Create a minimal HRRR analysis dataset for testing."""
    times = pd.date_range("2024-01-01", periods=4, freq="1h")
    x = np.arange(100)
    y = np.arange(100)

    ds = xr.Dataset(
        {
            "temperature_2m": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))).astype(np.float32),
            ),
            "precipitation_surface": (
                ["time", "y", "x"],
                rng.standard_normal((len(times), len(y), len(x))).astype(np.float32),
            ),
        },
        coords={
            "time": times,
            "y": y,
            "x": x,
        },
    )
    return ds


def test_check_analysis_recent_nans_no_nans(hrrr_analysis_dataset: xr.Dataset) -> None:
    """Test that check_analysis_recent_nans passes when there are no NaN values."""
    result = validators.check_analysis_recent_nans(hrrr_analysis_dataset)
    assert result.passed
    assert "within acceptable limit" in result.message


def test_check_analysis_recent_nans_with_excessive_nans(
    hrrr_analysis_dataset: xr.Dataset,
) -> None:
    """Test that check_analysis_recent_nans fails when there are excessive NaN values."""
    ds = hrrr_analysis_dataset.copy(deep=True)
    ds["temperature_2m"].values[-1, :50, :] = np.nan

    result = validators.check_analysis_recent_nans(ds, max_nan_percent=0.5)
    assert not result.passed
    assert "Excessive NaN values found" in result.message
    assert "temperature_2m" in result.message


def test_check_data_is_current(hrrr_analysis_dataset: xr.Dataset) -> None:
    """Test that check_data_is_current passes when data is recent."""
    ds = hrrr_analysis_dataset.copy(deep=True)
    ds = ds.assign_coords(
        time=pd.date_range(
            pd.Timestamp.now() - pd.Timedelta("3h"), periods=4, freq="1h"
        )
    )

    result = validators.check_data_is_current(ds)
    assert result.passed
    assert "Data is current" in result.message


def test_check_data_is_current_fails_when_old(
    hrrr_analysis_dataset: xr.Dataset,
) -> None:
    """Test that check_data_is_current fails when data is too old."""
    ds = hrrr_analysis_dataset.copy(deep=True)
    ds = ds.assign_coords(
        time=pd.date_range(
            pd.Timestamp.now() - pd.Timedelta("10h"), periods=4, freq="1h"
        )
    )

    result = validators.check_data_is_current(ds)
    assert not result.passed
    assert "Latest time is" in result.message
