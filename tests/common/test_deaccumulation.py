from typing import Final

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.deaccumulation import (
    deaccumulate_to_rates_inplace,
)

SECONDS_PER_HOUR: Final[int] = 60 * 60


def test_deaccumulate_higher_dimensional() -> None:
    # Create test data
    reset_frequency = pd.Timedelta(hours=6)
    init_time = pd.Timestamp("2024-01-01T06:00")
    lead_times = pd.timedelta_range("0h", "12h", freq="3h")

    three_hr_at_1mm = 3.0 * SECONDS_PER_HOUR
    six_hr_at_1mm = 6.0 * SECONDS_PER_HOUR

    # Create a 2D array with 2 ensemble members to test parallelization
    data = np.array(
        [
            # First ensemble member - rate=1.0
            [np.nan, three_hr_at_1mm, six_hr_at_1mm, three_hr_at_1mm, six_hr_at_1mm],
            # Second ensemble member - rate=1.0
            [np.nan, three_hr_at_1mm, six_hr_at_1mm, three_hr_at_1mm, six_hr_at_1mm],
            # Third ensemble member - rate=2.0
            [
                np.nan,
                2 * three_hr_at_1mm,
                2 * six_hr_at_1mm,
                2 * three_hr_at_1mm,
                2 * six_hr_at_1mm,
            ],
        ]
    )
    data = np.stack([data, 2 * data])
    # Add new dims of > len(1) to trailing dimensions
    data = np.stack([data, data], axis=-1)
    data = np.stack([data, data], axis=-1)

    data_array = xr.DataArray(
        data,  # Shape will be (2, 3, 5, 1, 1) for (init_time, ensemble_member, lead_time, lat, lon)
        coords={
            "init_time": [init_time, init_time + pd.Timedelta(hours=6)],
            "ensemble_member": [0, 1, 2],
            "lead_time": lead_times,
        },
        dims=["init_time", "ensemble_member", "lead_time", "latitude", "longitude"],
        attrs={"units": "mm/s"},
    )

    # Expected rates after deaccumulation
    expected = np.array(
        [
            [np.nan, 1.0, 1.0, 1.0, 1.0],  # First member
            [np.nan, 1.0, 1.0, 1.0, 1.0],  # Second member
            [np.nan, 2.0, 2.0, 2.0, 2.0],  # Third member
        ]
    )
    expected = np.stack([expected, 2 * expected])
    expected = np.stack([expected, expected], axis=-1)
    expected = np.stack([expected, expected], axis=-1)

    result = deaccumulate_to_rates_inplace(
        data_array, dim="lead_time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_1d_3_and_6_hour_normal_cases() -> None:
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    # These values will have large accumulations going in and output rates that in the single digits
    values = [
        # 3 hourly step:
        {"lt": 0, "in": np.nan, "out": np.nan},  # no deaccum on first step
        {"lt": 3, "in": 4 * sec * 3, "out": 4.0},  # standard 3h case
        {"lt": 6, "in": 4 * sec * 3, "out": 0.0},  # no new accumulation between 3h and 6h steps
        {"lt": 9, "in": 0 * sec * 3, "out": 0.0},  # no new accumulation between 6h and 3h steps
        {"lt": 12, "in": 2 * sec * 3, "out": 2.0}, # 0 mm accumulated in first 3 hours, 2 mm accumulated in next 3 hours
        # Test transition from 3h to 6h accumulation
        # 6 hourly step:
        {"lt": 18, "in": 3 * sec * 6, "out": 3.0},  # standard 6h case following a 3h step
        {"lt": 24, "in": 7 * sec * 6, "out": 7.0},  # standard 6h case
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array, dim="lead_time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_1d_hourly_6h_reset_normal_cases() -> None:
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    # These values will have large accumulations going in and output rates that in the single digits
    values = [
        # 1 hourly step:
        {"lt": 0, "in": np.nan, "out": np.nan},  # no deaccum on first step
        {"lt": 1, "in": 4 * sec, "out": 4.0},  # standard 1h case
        {"lt": 2, "in": 4 * sec, "out": 0.0},  # no new accumulation between 1h and 2h steps
        {"lt": 3, "in": 5 * sec, "out": 1.0},  # new accumulation between 2h and 3h steps
        {"lt": 4, "in": 7 * sec, "out": 2.0},  # 2 mm/s new accumulation
        {"lt": 5, "in": 7 * sec, "out": 0.0},  # no new accumulation
        {"lt": 6, "in": 7 * sec, "out": 0.0},  # no new accumulation
        {"lt": 7, "in": 7 * sec, "out": 7.0},  # 7 mm new accumulation after 6h reset
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array, dim="lead_time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_1d_hourly_3hourly_6hourly_6h_reset_normal_cases() -> None:
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    # These values will have large accumulations going in and output rates that in the single digits
    values = [
        # 1 hourly step:
        {"lt": 0, "in": np.nan, "out": np.nan},  # no deaccum on first step
        {"lt": 1, "in": 4 * sec, "out": 4.0},  # standard 1h case
        {"lt": 2, "in": 4 * sec, "out": 0.0},  # no new accumulation between 1h and 2h steps
        {"lt": 3, "in": 5 * sec, "out": 1.0},  # new accumulation between 2h and 3h steps
        {"lt": 4, "in": 7 * sec, "out": 2.0},  # 2 mm/s new accumulation
        {"lt": 5, "in": 7 * sec, "out": 0.0},  # no new accumulation
        {"lt": 6, "in": 7 * sec, "out": 0.0},  # no new accumulation
        # 3 hourly step:
        {"lt": 9, "in": 7 * sec * 3, "out": 7.0},  # 7 mm/s new accumulation over 3 hours after 6h reset
        {"lt": 12, "in": 8 * sec * 3, "out": 1.0},  # 1 mm/s new accumulation over 3 hours
        {"lt": 15, "in": 8 * sec * 3, "out": 8.0},  # 8 mm/s new accumulation over 3 hours
        {"lt": 18, "in": 9 * sec * 3, "out": 1.0},  # 1 mm/s new accumulation over 3 hours
        # 6 hourly step:
        {"lt": 24, "in": 3 * sec * 6, "out": 3.0},  # 3 mm/s new accumulation over 6 hours
        {"lt": 30, "in": 0 * sec * 6, "out": 0.0},  # no new accumulation over 6 hours
        {"lt": 36, "in": 5 * sec * 6, "out": 5.0},  # no new accumulation over 6 hours
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array, dim="lead_time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_1d_hourly_6h_reset_missing_values() -> None:
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    # These values will have large accumulations going in and output rates that in the single digits
    values = [
        # 1 hourly step:
        {"lt": 0, "in": np.nan, "out": np.nan},  # no deaccum on first step
        {"lt": 1, "in": np.nan, "out": np.nan},  # missing value
        {"lt": 2, "in": 4 * sec, "out": np.nan},  # missing value in previous step
        {"lt": 3, "in": 5 * sec, "out": 1.0},  # enough data to deaccum
        {"lt": 4, "in": np.nan, "out": np.nan},  # missing again
        {"lt": 5, "in": 7 * sec, "out": np.nan},  # missing value in previous step
        {"lt": 6, "in": np.nan, "out": np.nan},
        {"lt": 7, "in": 3 * sec, "out": 3.0},  # 3 mm new accumulation after 6h reset even though reset step was nan
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array, dim="lead_time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_1d_3_and_6_hour_small_accumulation_decreases() -> None:
    reset_frequency = pd.Timedelta(hours=6)

    values = [
        # 3 hourly step:
        {"lt": 0, "in": 0., "out": 0.},
        {"lt": 3, "in": 4. , "out": 4.0 / (3 * SECONDS_PER_HOUR)},
        {"lt": 6, "in": 3.9, "out": 0.0},  # small negative accumulation clamped to 0
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    with pytest.raises(
        ValueError, match="Over 5% \\(1 total\\) values were clamped to 0"
    ):
        deaccumulate_to_rates_inplace(
            data_array, dim="lead_time", reset_frequency=reset_frequency
        )


def test_deaccumulate_1d_3_and_6_hour_large_accumulation_decreases() -> None:
    reset_frequency = pd.Timedelta(hours=6)

    values = [
        {"lt": 0, "in": np.nan, "out": np.nan},
        {"lt": 3, "in": 2, "out": 2.0 / (3 * SECONDS_PER_HOUR)},
        {"lt": 6, "in": 1.7, "out": np.nan},  # negative accumulation too large, set to NaN
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    with pytest.raises(ValueError, match="Found 1 values below threshold"):
        deaccumulate_to_rates_inplace(
            data_array, dim="lead_time", reset_frequency=reset_frequency
        )


def test_deaccumulate_1d_time_dim_3_and_6_hour_normal_cases() -> None:
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    # These values will have large accumulations going in and output rates that in the single digits
    values = [
        # 3 hourly step:
        {"lt": 0, "in": np.nan, "out": np.nan},  # no deaccum on first step
        {"lt": 3, "in": 4 * sec * 3, "out": 4.0},  # standard 3h case
        {"lt": 6, "in": 4 * sec * 3, "out": 0.0},  # no new accumulation between 3h and 6h steps
        {"lt": 9, "in": 0 * sec * 3, "out": 0.0},  # no new accumulation between 6h and 3h steps
        {"lt": 12, "in": 2 * sec * 3, "out": 2.0}, # 0 mm accumulated in first 3 hours, 2 mm accumulated in next 3 hours
        # Test transition from 3h to 6h accumulation
        # 6 hourly step:
        {"lt": 18, "in": 3 * sec * 6, "out": 3.0},  # standard 6h case following a 3h step
        {"lt": 24, "in": 7 * sec * 6, "out": 7.0},  # standard 6h case
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    # Start time at 03:00 tests reseting works regardless of the start time
    times = pd.Timestamp("2000-01-01T00:00") + lead_times
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"time": times},
        dims=["time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array, dim="time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_skip_steps_all_false() -> None:
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    # These values will have large accumulations going in and output rates that in the single digits
    values = [
        # 3 hourly step:
        {"lt": 0, "in": np.nan, "out": np.nan},  # no deaccum on first step
        {"lt": 3, "in": 4 * sec * 3, "out": 4.0},  # standard 3h case
        {"lt": 6, "in": 4 * sec * 3, "out": 0.0},  # no new accumulation between 3h and 6h steps
        {"lt": 9, "in": 0 * sec * 3, "out": 0.0},  # no new accumulation between 6h and 3h steps
        {"lt": 12, "in": 2 * sec * 3, "out": 2.0}, # 0 mm accumulated in first 3 hours, 2 mm accumulated in next 3 hours
        # Test transition from 3h to 6h accumulation
        # 6 hourly step:
        {"lt": 18, "in": 3 * sec * 6, "out": 3.0},  # standard 6h case following a 3h step
        {"lt": 24, "in": 7 * sec * 6, "out": 7.0},  # standard 6h case
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    # Start time at 03:00 tests reseting works regardless of the start time
    times = pd.Timestamp("2000-01-01T00:00") + lead_times
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"time": times},
        dims=["time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="time",
        reset_frequency=reset_frequency,
        skip_step=np.zeros(len(times), dtype=np.bool),
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_1d_hourly_window_and_step() -> None:
    reset_frequency = pd.Timedelta(hours=1)
    sec = float(SECONDS_PER_HOUR)

    values = [
        {"lt": 0, "in": np.nan, "out": np.nan},  # no deaccum on first step
        {"lt": 1, "in": 4 * sec, "out": 4.0},
        {"lt": 2, "in": 0 * sec, "out": 0.0},
        {"lt": 3, "in": 5 * sec, "out": 5.0},
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array, dim="lead_time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_1d_skip_every_other_step() -> None:
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)
    nan = np.nan

    # These values will have large accumulations going in and output rates that in the single digits
    values = [
        # 3 hourly step with 6 hourly data
        {"lt": 0, "in": nan, "out": nan, "skip": False},  # no deaccum on first step
        {"lt": 3, "in": nan, "out": nan, "skip": True},  # skip step
        {"lt": 6, "in": 4 * sec * 6, "out": 4.0, "skip": False},  # 4 mm/s accumulation in last 6 hours
        {"lt": 9, "in": nan, "out": nan, "skip": True},  # skip step
        {"lt": 12, "in": 2 * sec * 6, "out": 2.0, "skip": False}, # 2 mm/s accumulation in last 6 hours
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    skip_step = np.array([step["skip"] for step in values])
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        skip_step=skip_step,
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_reset_frequency_equals_max_lead_time() -> None:
    """Test case where reset frequency equals the maximum lead time."""
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    values = [
        {"lt": 0, "in": np.nan, "out": np.nan},  # no deaccum on first step
        {"lt": 3, "in": 3 * sec * 3, "out": 3.0},  # 3h accumulation
        {"lt": 6, "in": 6 * sec * 6, "out": 9.0},  # 6h accumulation, resets after this step
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array, dim="lead_time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_first_step_non_nan_becomes_nan() -> None:
    """Test case where first step has non-nan value but must become nan (nothing to deaccumulate from)."""
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    values = [
        {"lt": 0, "in": 5.0, "out": np.nan},  # non-nan input becomes nan output (no previous value to deaccumulate from)
        {"lt": 3, "in": 8 * sec * 3, "out": 8.0},  # standard 3h case
        {"lt": 6, "in": 8 * sec * 3, "out": 0.0},  # no new accumulation between 3h and 6h steps
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array, dim="lead_time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)
