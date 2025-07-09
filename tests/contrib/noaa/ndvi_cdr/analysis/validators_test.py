import numpy as np
import pandas as pd
import xarray as xr
from pytest import MonkeyPatch

from reformatters.contrib.noaa.ndvi_cdr.ndvi_cdr.analysis.validators import (
    check_data_is_current,
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
