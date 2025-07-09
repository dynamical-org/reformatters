import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import MonkeyPatch

from reformatters.contrib.noaa.ndvi_cdr.ndvi_cdr.analysis.validators import (
    check_data_is_current,
    check_latest_ndvi_usable_nan_percentage,
)


def test_check_data_is_current_success(monkeypatch: MonkeyPatch) -> None:
    """Test passes when data is current within the last 4 days."""
    monkeypatch.setattr(pd.Timestamp, "now", lambda: pd.Timestamp("2024-01-05"))

    # Create data within the last 4 days
    times = pd.date_range("2024-01-01", periods=3, freq="1D")
    data = np.ones((3, 3, 3))
    ds = xr.Dataset(
        {"ndvi": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    result = check_data_is_current(ds)

    assert result.passed is True
    assert "Data found for the last 4 days" in result.message


def test_check_data_is_current_failure(monkeypatch: MonkeyPatch) -> None:
    """Test fails when no data is found within the last 4 days."""
    monkeypatch.setattr(pd.Timestamp, "now", lambda: pd.Timestamp("2024-01-10"))

    # Create data older than 4 days
    times = pd.date_range("2024-01-01", periods=3, freq="1D")
    data = np.ones((3, 3, 3))
    ds = xr.Dataset(
        {"ndvi": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    result = check_data_is_current(ds)

    assert result.passed is False
    assert "No data found for the last 4 days" in result.message


def test_check_data_is_current_success_with_nonzero_time(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test passes when data is current even when current time has non-zero hour."""
    monkeypatch.setattr(
        pd.Timestamp, "now", lambda: pd.Timestamp("2024-01-05T14:30:00")
    )

    # Create data within the last 4 days
    times = pd.date_range("2024-01-02", periods=3, freq="1D")
    data = np.ones((3, 3, 3))
    ds = xr.Dataset(
        {"ndvi": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    result = check_data_is_current(ds)

    assert result.passed is True
    assert "Data found for the last 4 days" in result.message


@pytest.mark.parametrize(
    "nan_count,expected_pass",
    [
        (93, True),  # 93% NaN, should pass
        (94, False),  # 94% NaN, should fail
        (95, False),  # 95% NaN, should fail
    ],
)
def test_check_latest_ndvi_usable_nan_percentage_threshold(
    nan_count: int, expected_pass: bool
) -> None:
    """Test validator at, below, and above the 94% NaN threshold."""
    times = pd.date_range("2024-01-01", periods=2, freq="1D")
    shape = (2, 10, 10)  # 100 values per time step
    ndvi_usable = np.ones(shape)
    ndvi_usable[-1, :, :] = 1
    ndvi_usable[-1].flat[:nan_count] = np.nan  # Set NaNs in last time step

    ds = xr.Dataset(
        {"ndvi_usable": (["time", "latitude", "longitude"], ndvi_usable)},
        coords={"time": times, "latitude": np.arange(10), "longitude": np.arange(10)},
    )

    result = check_latest_ndvi_usable_nan_percentage(ds)
    assert result.passed is expected_pass

    if expected_pass:
        assert "expected nan percentage" in result.message
    else:
        assert "high nan percentage" in result.message
