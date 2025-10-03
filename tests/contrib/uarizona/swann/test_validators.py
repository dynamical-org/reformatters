import numpy as np
import pandas as pd
import xarray as xr
from pytest import MonkeyPatch

from reformatters.contrib.uarizona.swann.analysis.validators import (
    check_data_is_current,
    check_latest_time_nans,
)


def test_check_latest_time_nans_success() -> None:
    """Test passes when most recent time step has low NaN percentage."""
    times = pd.date_range("2024-01-01", periods=3, freq="6h")

    # Create data where only the last time step has NaNs
    data = np.ones((3, 3, 3))

    ds = xr.Dataset(
        {"snow_water_equivalent": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    result = check_latest_time_nans(ds)

    assert result.passed is True
    assert "check_latest_time_nans" in result.message
    assert "acceptable NaN percentages" in result.message


def test_check_latest_time_nans_failure() -> None:
    """Test passes when most recent time step has low NaN percentage."""
    times = pd.date_range("2024-01-01", periods=3, freq="6h")

    data = np.ones((3, 3, 3))
    data[-1, :, :] = np.nan  # Last time step all NaN

    ds = xr.Dataset(
        {"snow_water_equivalent": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    result = check_latest_time_nans(ds)

    assert result.passed is False
    assert "check_latest_time_nans" in result.message
    assert "found excessive NaN values" in result.message


def test_check_data_is_current_success(monkeypatch: MonkeyPatch) -> None:
    """Test passes when data is current."""
    monkeypatch.setattr(pd.Timestamp, "now", lambda: pd.Timestamp("2024-01-05"))

    times = pd.date_range("2024-01-01", periods=3, freq="1D")
    data = np.ones((3, 3, 3))
    ds = xr.Dataset(
        {"snow_water_equivalent": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    result = check_data_is_current(ds)

    assert result.passed is True
    assert "Data found for the last" in result.message


def test_check_data_is_current_success_when_current_hour_is_nonzero(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test passes when data is current."""
    monkeypatch.setattr(
        pd.Timestamp, "now", lambda: pd.Timestamp("2024-01-05T01:00:00")
    )

    times = pd.date_range("2024-01-01", periods=3, freq="1D")
    data = np.ones((3, 3, 3))
    ds = xr.Dataset(
        {"snow_water_equivalent": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    result = check_data_is_current(ds)

    assert result.passed is True
    assert "Data found for the last" in result.message


def test_check_data_is_current_failure(monkeypatch: MonkeyPatch) -> None:
    """Test passes when data is current."""
    monkeypatch.setattr(pd.Timestamp, "now", lambda: pd.Timestamp("2024-01-09"))

    times = pd.date_range("2024-01-01", periods=3, freq="1D")
    data = np.ones((3, 3, 3))
    ds = xr.Dataset(
        {"snow_water_equivalent": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )

    result = check_data_is_current(ds)

    assert result.passed is False
    assert "No data found for the last" in result.message
