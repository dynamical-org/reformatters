import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr.storage

from reformatters.common import validation
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour.validators import (
    CheckForecastCompleteness,
)


def _context(ds: xr.Dataset) -> validation.ValidationContext:
    # The real template config's data_vars drive hour-0 skipping via
    # has_hour_0_values() (e.g. HRRR categorical vars).
    return validation.ValidationContext(
        store=zarr.storage.MemoryStore(),
        ds=ds,
        append_dim="init_time",
        data_vars=NoaaHrrrForecast48HourTemplateConfig().data_vars,
    )


@pytest.fixture
def hrrr_forecast_dataset(rng: np.random.Generator) -> xr.Dataset:
    """Create a minimal HRRR forecast dataset for testing."""
    init_times = pd.date_range("2024-01-01", periods=4, freq="6h")
    lead_times = pd.timedelta_range("0h", "48h", freq="1h")
    x = np.arange(100)
    y = np.arange(100)

    ds = xr.Dataset(
        {
            "temperature_2m": (
                ["init_time", "lead_time", "y", "x"],
                rng.standard_normal(
                    (len(init_times), len(lead_times), len(y), len(x))
                ).astype(np.float32),
                {"step_type": "instant"},
            ),
            "precipitation_surface": (
                ["init_time", "lead_time", "y", "x"],
                rng.standard_normal(
                    (len(init_times), len(lead_times), len(y), len(x))
                ).astype(np.float32),
                {"step_type": "accum"},
            ),
            "categorical_rain_surface": (
                ["init_time", "lead_time", "y", "x"],
                rng.standard_normal(
                    (len(init_times), len(lead_times), len(y), len(x))
                ).astype(np.float32),
                {"step_type": "instant"},
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


def test_check_recent_nans_no_nans(hrrr_forecast_dataset: xr.Dataset) -> None:
    """Default (strict) passes when there are no NaNs."""
    result = validation.CheckRecentNans().check(_context(hrrr_forecast_dataset))
    assert result.passed


def test_check_recent_nans_with_excessive_nans(
    hrrr_forecast_dataset: xr.Dataset,
) -> None:
    """Fails when an instant var has excessive NaNs."""
    ds = hrrr_forecast_dataset.copy(deep=True)
    # NaN the entire latest init_time so any random_points sample lands on it
    ds["temperature_2m"].values[-1, :, :, :] = np.nan

    result = validation.CheckRecentNans(max_nan_fraction=0.005).check(_context(ds))
    assert not result.passed
    assert "Excessive NaN fraction" in result.message
    assert "temperature_2m" in result.message


def test_check_recent_nans_skips_lead_time_0_for_accumulations(
    hrrr_forecast_dataset: xr.Dataset,
) -> None:
    """lead_time=0 is skipped for accumulation variables (step_type=accum)."""
    ds = hrrr_forecast_dataset.copy(deep=True)
    ds["precipitation_surface"].values[-1, 0, :, :] = np.nan

    result = validation.CheckRecentNans(max_nan_fraction=0.005).check(_context(ds))
    assert result.passed


def test_check_recent_nans_skips_lead_time_0_for_expected_hour_0_nan_vars(
    hrrr_forecast_dataset: xr.Dataset,
) -> None:
    """HRRR categorical vars (step_type=instant but no hour 0) skip lead_time=0."""
    ds = hrrr_forecast_dataset.copy(deep=True)
    ds["categorical_rain_surface"].values[-1, 0, :, :] = np.nan

    result = validation.CheckRecentNans(max_nan_fraction=0.005).check(_context(ds))
    assert result.passed


def test_check_forecast_completeness_passes(rng: np.random.Generator) -> None:
    """CheckForecastCompleteness passes when expected lead times are present."""
    init_times = pd.date_range("2024-01-01", periods=4, freq="6h")
    lead_times = pd.timedelta_range("0h", "48h", freq="1h")

    ds = xr.Dataset(
        {
            "temperature_2m": (
                ["init_time", "lead_time", "y", "x"],
                rng.standard_normal((len(init_times), len(lead_times), 5, 5)).astype(
                    np.float32
                ),
                {"step_type": "instant"},
            ),
        },
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "y": np.arange(5),
            "x": np.arange(5),
        },
    )

    result = CheckForecastCompleteness().check(_context(ds))
    assert result.passed
