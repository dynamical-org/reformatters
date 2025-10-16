from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation


@pytest.fixture
def forecast_dataset() -> xr.Dataset:
    """Create a mock forecast dataset for testing."""
    init_times = pd.date_range("2024-01-01", periods=5, freq="6h")
    lead_times = pd.timedelta_range(start="0h", end="240h", freq="6h")
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)

    ds = xr.Dataset(
        {
            "temperature": (
                ["init_time", "lead_time", "latitude", "longitude"],
                np.random.randn(len(init_times), len(lead_times), len(lats), len(lons)),
            ),
            "precipitation": (
                ["init_time", "lead_time", "latitude", "longitude"],
                np.random.randn(len(init_times), len(lead_times), len(lats), len(lons)),
            ),
        },
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


@pytest.fixture
def analysis_dataset() -> xr.Dataset:
    """Create a mock analysis dataset for testing."""
    times = pd.date_range("2024-01-01", periods=48, freq="1h")
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)

    ds = xr.Dataset(
        {
            "temperature": (
                ["time", "latitude", "longitude"],
                np.random.randn(len(times), len(lats), len(lons)),
            ),
            "humidity": (
                ["time", "latitude", "longitude"],
                np.random.randn(len(times), len(lats), len(lons)),
            ),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


def test_check_forecast_current_data_passes(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """Test that check_forecast_current_data passes when recent data exists."""
    now = pd.Timestamp("2024-01-01 18:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda: now)

    result = validation.check_forecast_current_data(forecast_dataset)

    assert result.passed
    assert "Data found for the latest day" in result.message


def test_check_forecast_current_data_fails(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """Test that check_forecast_current_data fails when no recent data exists."""
    now = pd.Timestamp("2024-01-10")
    monkeypatch.setattr("pandas.Timestamp.now", lambda: now)

    result = validation.check_forecast_current_data(forecast_dataset)

    assert not result.passed
    assert "No data found for the latest day" in result.message


def test_check_forecast_recent_nans_passes(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """Test that check_forecast_recent_nans passes when NaN percentage is acceptable."""
    now = pd.Timestamp("2024-01-01 18:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda: now)

    result = validation.check_forecast_recent_nans(forecast_dataset)

    assert result.passed
    assert "acceptable NaN percentages" in result.message


def test_check_forecast_recent_nans_fails(
    monkeypatch: pytest.MonkeyPatch, forecast_dataset: xr.Dataset
) -> None:
    """Test that check_forecast_recent_nans fails when NaN percentage is too high."""
    now = pd.Timestamp("2024-01-01 18:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda: now)

    # Add excessive NaNs to the latest init_time
    forecast_dataset["temperature"].loc[
        {"init_time": forecast_dataset.init_time[-1]}
    ] = np.nan

    result = validation.check_forecast_recent_nans(
        forecast_dataset, max_nan_percentage=10
    )

    assert not result.passed
    assert "Excessive NaN values found" in result.message
    assert "temperature" in result.message


def test_check_analysis_current_data_passes(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_current_data passes when recent data exists."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    result = validation.check_analysis_current_data(analysis_dataset)

    assert result.passed
    assert "Data found within" in result.message


def test_check_analysis_current_data_fails(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_current_data fails when no recent data exists."""
    now = pd.Timestamp("2024-01-10")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    result = validation.check_analysis_current_data(analysis_dataset)

    assert not result.passed
    assert "No data found within" in result.message


def test_check_analysis_current_data_custom_delay(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_current_data respects custom max_expected_delay."""
    # Dataset ends at 2024-01-02 23:00:00
    # Check at 2024-01-03 12:00:00 (12.5 hours after last data)
    now = pd.Timestamp("2024-01-03 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    # Should fail with default 12 hour delay (last data is 12.5 hours ago)
    result = validation.check_analysis_current_data(analysis_dataset)
    assert not result.passed

    # Should pass with 48 hour delay
    result = validation.check_analysis_current_data(
        analysis_dataset, max_expected_delay=timedelta(hours=48)
    )
    assert result.passed


def test_check_analysis_recent_nans_passes(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans passes when NaN percentage is acceptable."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    result = validation.check_analysis_recent_nans(analysis_dataset)

    assert result.passed
    assert "acceptable NaN percentages" in result.message


def test_check_analysis_recent_nans_fails(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans fails when NaN percentage is too high."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    # Mock random sampling to return coordinates within dataset bounds
    # First call returns lon=0, second call returns lat=0
    call_count = [0]

    def mock_uniform(low: float, high: float) -> float:
        call_count[0] += 1
        if call_count[0] == 1:  # longitude
            return 0.0
        else:  # latitude
            return 2.0  # Use positive value so slice(2, 0) selects data

    monkeypatch.setattr("numpy.random.uniform", mock_uniform)

    # Set all recent data to NaN to ensure the random sample will catch it
    analysis_dataset["temperature"].loc[{"time": slice("2024-01-02", None)}] = np.nan

    result = validation.check_analysis_recent_nans(
        analysis_dataset,
        max_expected_delay=timedelta(hours=12),
        max_nan_percentage=5,
    )

    assert not result.passed
    assert "Excessive NaN values found" in result.message
    assert "temperature" in result.message


def test_check_analysis_recent_nans_custom_parameters(
    monkeypatch: pytest.MonkeyPatch, analysis_dataset: xr.Dataset
) -> None:
    """Test that check_analysis_recent_nans respects custom parameters."""
    now = pd.Timestamp("2024-01-02 12:00:00")
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: now)

    # Mock random sampling to return coordinates within dataset bounds
    call_count = [0]

    def mock_uniform(low: float, high: float) -> float:
        call_count[0] += 1
        if call_count[0] % 2 == 1:  # longitude (odd calls)
            return 0.0
        else:  # latitude (even calls)
            return 2.0

    monkeypatch.setattr("numpy.random.uniform", mock_uniform)

    # Set all recent data to NaN to ensure the random sample will catch it
    recent_slice = {"time": slice("2024-01-02", None)}
    analysis_dataset["temperature"].loc[recent_slice] = np.nan

    # Should fail with 5% threshold
    result = validation.check_analysis_recent_nans(
        analysis_dataset,
        max_expected_delay=timedelta(hours=12),
        max_nan_percentage=5,
    )
    assert not result.passed

    # Should pass with 100% threshold
    result = validation.check_analysis_recent_nans(
        analysis_dataset,
        max_expected_delay=timedelta(hours=12),
        max_nan_percentage=100,
    )
    assert result.passed


def test_validation_result_model() -> None:
    """Test ValidationResult pydantic model."""
    result = validation.ValidationResult(passed=True, message="Test passed")
    assert result.passed
    assert result.message == "Test passed"

    result = validation.ValidationResult(passed=False, message="Test failed")
    assert not result.passed
    assert result.message == "Test failed"
