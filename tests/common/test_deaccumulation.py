from itertools import pairwise
from typing import Final

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.deaccumulation import (
    PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
    RADIATION_INVALID_BELOW_THRESHOLD,
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
        data,  # Shape will be (2, 3, 5, 2, 2) for (init_time, ensemble_member, lead_time, lat, lon)
        coords={
            "init_time": [init_time, init_time + pd.Timedelta(hours=6)],
            "ensemble_member": [0, 1, 2],
            "lead_time": lead_times,
        },
        dims=["init_time", "ensemble_member", "lead_time", "latitude", "longitude"],
        attrs={"units": "mm s-1"},
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
        attrs={"units": "mm s-1"},
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
        {"lt": 4, "in": 7 * sec, "out": 2.0},  # 2 mm s-1 new accumulation
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
        attrs={"units": "mm s-1"},
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
        {"lt": 4, "in": 7 * sec, "out": 2.0},  # 2 mm s-1 new accumulation
        {"lt": 5, "in": 7 * sec, "out": 0.0},  # no new accumulation
        {"lt": 6, "in": 7 * sec, "out": 0.0},  # no new accumulation
        # 3 hourly step:
        {"lt": 9, "in": 7 * sec * 3, "out": 7.0},  # 7 mm s-1 new accumulation over 3 hours after 6h reset
        {"lt": 12, "in": 8 * sec * 3, "out": 1.0},  # 1 mm s-1 new accumulation over 3 hours
        {"lt": 15, "in": 8 * sec * 3, "out": 8.0},  # 8 mm s-1 new accumulation over 3 hours
        {"lt": 18, "in": 9 * sec * 3, "out": 1.0},  # 1 mm s-1 new accumulation over 3 hours
        # 6 hourly step:
        {"lt": 24, "in": 3 * sec * 6, "out": 3.0},  # 3 mm s-1 new accumulation over 6 hours
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
        attrs={"units": "mm s-1"},
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
        attrs={"units": "mm s-1"},
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
        attrs={"units": "mm s-1"},
    )

    with pytest.raises(
        ValueError, match=r"Over 5% \(1 total, 33.3%\) values were clamped to 0"
    ):
        deaccumulate_to_rates_inplace(
            data_array, dim="lead_time", reset_frequency=reset_frequency
        )


def test_deaccumulate_expected_clamp_fraction_suppresses_error() -> None:
    """When expected_clamp_fraction covers the actual clamp fraction, no error is raised."""
    reset_frequency = pd.Timedelta(hours=6)

    values = [
        {"lt": 0, "in": np.nan, "out": np.nan},
        {"lt": 3, "in": 4. , "out": 4.0 / (3 * SECONDS_PER_HOUR)},
        {"lt": 6, "in": 3.9, "out": 0.0},  # small negative accumulation clamped to 0
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        expected_clamp_fraction=0.34,
    )
    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_expected_clamp_fraction_still_raises_when_exceeded() -> None:
    """When actual clamp fraction exceeds expected_clamp_fraction, error is still raised."""
    reset_frequency = pd.Timedelta(hours=6)

    values = [
        {"lt": 0, "in": 0., "out": 0.},
        {"lt": 3, "in": 4. , "out": 4.0 / (3 * SECONDS_PER_HOUR)},
        {"lt": 6, "in": 3.9, "out": 0.0},  # small negative accumulation clamped to 0 (1 of 3 = 33%)
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )

    with pytest.raises(
        ValueError, match=r"Over 10% \(1 total, 33.3%\) values were clamped to 0"
    ):
        deaccumulate_to_rates_inplace(
            data_array,
            dim="lead_time",
            reset_frequency=reset_frequency,
            expected_clamp_fraction=0.10,
        )


def test_deaccumulate_1d_3_and_6_hour_large_accumulation_decreases() -> None:
    reset_frequency = pd.Timedelta(hours=6)

    values = [
        {"lt": 0, "in": np.nan, "out": np.nan},
        {"lt": 3, "in": 2, "out": 2.0 / (3 * SECONDS_PER_HOUR)},
        {"lt": 6, "in": 1, "out": np.nan},  # negative accumulation too large, set to NaN
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )

    with pytest.raises(ValueError, match=r"Found 1 values .* below threshold"):
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
        attrs={"units": "mm s-1"},
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
        attrs={"units": "mm s-1"},
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
        attrs={"units": "mm s-1"},
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
        {"lt": 6, "in": 4 * sec * 6, "out": 4.0, "skip": False},  # 4 mm s-1 accumulation in last 6 hours
        {"lt": 9, "in": nan, "out": nan, "skip": True},  # skip step
        {"lt": 12, "in": 2 * sec * 6, "out": 2.0, "skip": False}, # 2 mm s-1 accumulation in last 6 hours
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    skip_step = np.array([step["skip"] for step in values])
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
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
        attrs={"units": "mm s-1"},
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
        attrs={"units": "mm s-1"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array, dim="lead_time", reset_frequency=reset_frequency
    )

    np.testing.assert_equal(result.values, expected)


@pytest.mark.parametrize(
    ("threshold", "should_raise"),
    [
        (-1.0, False),  # permissive threshold allows clamping
        (
            PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
            True,
        ),
    ],
)
def test_custom_deaccumulate_invalid_threshold_rate(
    threshold: float, should_raise: bool
) -> None:
    """Test that invalid_below_threshold_rate controls clamping vs raising."""
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    lt_6_in_val = 1.7

    # With the more permissive threshold, the value should be clamped to 0
    # With the default threshold, the value should be set to NaN (and we expect to raise an error)
    lt_6_out_val = np.nan if should_raise else 0.0

    values = [
        {"lt": 0, "in": np.nan, "out": np.nan},
        {"lt": 3, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 6, "in": lt_6_in_val * sec * 3, "out": lt_6_out_val},
        {"lt": 9, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 12, "in": 4 * sec * 3, "out": 2.0},
        {"lt": 15, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 18, "in": 4 * sec * 3, "out": 2.0},
        {"lt": 21, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 24, "in": 4 * sec * 3, "out": 2.0},
        {"lt": 27, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 30, "in": 4 * sec * 3, "out": 2.0},
        {"lt": 33, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 36, "in": 4 * sec * 3, "out": 2.0},
        {"lt": 39, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 42, "in": 4 * sec * 3, "out": 2.0},
        {"lt": 45, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 48, "in": 4 * sec * 3, "out": 2.0},
        {"lt": 51, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 54, "in": 4 * sec * 3, "out": 2.0},
        {"lt": 57, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 60, "in": 4 * sec * 3, "out": 2.0},
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )

    if should_raise:
        with pytest.raises(ValueError, match=r"Found 1 values .* below threshold"):
            deaccumulate_to_rates_inplace(
                data_array,
                dim="lead_time",
                reset_frequency=reset_frequency,
                invalid_below_threshold_rate=threshold,
            )
    else:
        result = deaccumulate_to_rates_inplace(
            data_array,
            dim="lead_time",
            reset_frequency=reset_frequency,
            invalid_below_threshold_rate=threshold,
        )
        np.testing.assert_equal(result.values, expected)


def test_deaccumulate_non_reset_aligned_first_step() -> None:
    """Test deaccumulation when first step is NOT aligned with reset frequency.

    This tests the shard boundary case where processing starts at e.g. hour 23
    instead of hour 0/6/12/18. The first step's accumulation value should be
    used as the baseline for subsequent calculations.

    Scenario with 6-hour reset frequency:
    - Hour 23: accumulation = 2.0 (accumulated since hour 18 reset)
    - Hour 0: accumulation = 5.0 (accumulated since hour 18, just before reset)
    - Hour 1: accumulation = 1.0 (fresh window since hour 0 reset)

    Expected rates:
    - Hour 23 → NaN (can't compute without hour 22 value)
    - Hour 0: rate = (5.0 - 2.0) / 3600 = 3.0/3600 per second
    - Hour 1: rate = 1.0 / 3600 per second (fresh window, baseline is 0)
    """
    reset_frequency = pd.Timedelta(hours=6)

    # Times starting at hour 23 (NOT a reset point, last reset was at 18:00)
    times = pd.DatetimeIndex(
        [
            "2024-01-01T23:00",  # 5h since reset, NOT a reset point
            "2024-01-02T00:00",  # 6h since reset, IS a reset point
            "2024-01-02T01:00",  # 1h since new reset at 00:00
        ]
    )

    # Accumulation values (accumulated since their respective reset points)
    # Hour 23: 2.0 accumulated since 18:00
    # Hour 0: 5.0 accumulated since 18:00 (3.0 more than hour 23)
    # Hour 1: 1.0 accumulated since 00:00 reset
    accumulations = np.array([2.0, 5.0, 1.0], dtype=np.float32)

    # Expected rates (per second)
    # Hour 23: NaN (can't compute)
    # Hour 0: (5.0 - 2.0) / 3600 seconds = 3.0 / 3600
    # Hour 1: 1.0 / 3600 seconds (fresh window)
    expected_rates = np.array([np.nan, 3.0 / 3600, 1.0 / 3600], dtype=np.float32)

    data_array = xr.DataArray(
        accumulations,
        coords={"time": times},
        dims=["time"],
        attrs={"units": "mm s-1"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="time",
        reset_frequency=reset_frequency,
    )

    np.testing.assert_allclose(
        result.values,
        expected_rates,
        rtol=1e-6,
        equal_nan=True,
    )


def test_deaccumulate_non_reset_aligned_first_step_with_lead_time() -> None:
    """Test non-reset-aligned first step using lead_time (timedelta) dimension."""
    reset_frequency = pd.Timedelta(hours=6)

    # Lead times starting at 5h (NOT a reset point for 6h reset)
    lead_times = pd.to_timedelta(["5h", "6h", "7h"])

    # Accumulation values
    # 5h: 2.0 accumulated since 0h
    # 6h: 5.0 accumulated since 0h (reset happens at 6h)
    # 7h: 1.0 accumulated since 6h reset
    accumulations = np.array([2.0, 5.0, 1.0], dtype=np.float32)

    # Expected rates (per second)
    # 5h: NaN (first step)
    # 6h: (5.0 - 2.0) / 3600 = 3.0 / 3600
    # 7h: 1.0 / 3600 (fresh window after reset)
    expected_rates = np.array([np.nan, 3.0 / 3600, 1.0 / 3600], dtype=np.float32)

    data_array = xr.DataArray(
        accumulations,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
    )

    np.testing.assert_allclose(
        result.values,
        expected_rates,
        rtol=1e-6,
        equal_nan=True,
    )


def test_deaccumulate_non_reset_aligned_multidimensional() -> None:
    """Test non-reset-aligned first step with multiple spatial dimensions."""
    reset_frequency = pd.Timedelta(hours=6)

    # Times starting at hour 23 (NOT a reset point)
    times = pd.DatetimeIndex(
        [
            "2024-01-01T23:00",
            "2024-01-02T00:00",
            "2024-01-02T01:00",
        ]
    )

    # 2x2 spatial grid, time in middle dimension
    # Shape: (2, 3, 2) = (y, time, x)
    accumulations = np.array(
        [
            [[2.0, 4.0], [5.0, 10.0], [1.0, 2.0]],  # y=0
            [[1.0, 2.0], [2.5, 5.0], [0.5, 1.0]],  # y=1
        ],
        dtype=np.float32,
    )

    # Expected rates - same pattern scaled by spatial location
    expected_rates = np.array(
        [
            [[np.nan, np.nan], [3.0 / 3600, 6.0 / 3600], [1.0 / 3600, 2.0 / 3600]],
            [[np.nan, np.nan], [1.5 / 3600, 3.0 / 3600], [0.5 / 3600, 1.0 / 3600]],
        ],
        dtype=np.float32,
    )

    data_array = xr.DataArray(
        accumulations,
        coords={"y": [0, 1], "time": times, "x": [0, 1]},
        dims=["y", "time", "x"],
        attrs={"units": "mm s-1"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="time",
        reset_frequency=reset_frequency,
    )

    np.testing.assert_allclose(
        result.values,
        expected_rates,
        rtol=1e-6,
        equal_nan=True,
    )


def test_deaccumulate_float64_input() -> None:
    """Test that float64 input arrays are supported."""
    reset_frequency = pd.Timedelta(hours=6)
    lead_times = pd.to_timedelta(["0h", "3h", "6h"])

    # Explicitly create float64 array
    accumulations = np.array([0.0, 3600.0 * 3, 3600.0 * 6], dtype=np.float64)
    assert accumulations.dtype == np.float64

    data_array = xr.DataArray(
        accumulations,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
    )

    # Should work and produce correct rates
    expected_rates = np.array([np.nan, 1.0, 1.0])
    np.testing.assert_allclose(
        result.values,
        expected_rates,
        rtol=1e-6,
        equal_nan=True,
    )


def test_deaccumulate_expected_invalid_fraction_suppresses_error() -> None:
    """When expected_invalid_fraction covers the actual invalid fraction, no error is raised."""
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    values = [
        {"lt": 0, "in": np.nan, "out": np.nan},
        {"lt": 3, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 6, "in": 1 * sec * 3, "out": np.nan},  # large negative → NaN (1 of 3 = 33%)
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)
    expected = np.array([step["out"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )

    # Without expected_invalid_fraction, this raises
    with pytest.raises(ValueError, match="below threshold"):
        deaccumulate_to_rates_inplace(
            data_array.copy(), dim="lead_time", reset_frequency=reset_frequency
        )

    # With sufficient expected_invalid_fraction, it succeeds
    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        expected_invalid_fraction=0.34,
    )
    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_expected_invalid_fraction_still_raises_when_exceeded() -> None:
    """When actual invalid fraction exceeds expected_invalid_fraction, error is still raised."""
    reset_frequency = pd.Timedelta(hours=6)
    sec = float(SECONDS_PER_HOUR)

    values = [
        {"lt": 0, "in": np.nan, "out": np.nan},
        {"lt": 3, "in": 2 * sec * 3, "out": 2.0},
        {"lt": 6, "in": 1 * sec * 3, "out": np.nan},  # large negative → NaN (1 of 3 = 33%)
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in values], unit="h")
    data = np.array([step["in"] for step in values], dtype=np.float32)

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )

    # expected_invalid_fraction too low → still raises
    with pytest.raises(ValueError, match=r"expected at most 10\.0%"):
        deaccumulate_to_rates_inplace(
            data_array,
            dim="lead_time",
            reset_frequency=reset_frequency,
            expected_invalid_fraction=0.10,
        )


def test_deaccumulate_running_mean_constant_rate() -> None:
    """A constant running-mean rate should deaccumulate to that same rate each step."""
    reset_frequency = pd.Timedelta.max
    lead_times = pd.to_timedelta(["0h", "1h", "2h", "3h"])

    # Constant 5 W m-2 averaged over every [0, t] window means every step saw 5 W m-2.
    values = np.array([0.0, 5.0, 5.0, 5.0], dtype=np.float32)
    expected = np.array([np.nan, 5.0, 5.0, 5.0], dtype=np.float32)

    data_array = xr.DataArray(
        values,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "W m-2"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        accumulation_type="running_mean",
    )

    np.testing.assert_allclose(result.values, expected, equal_nan=True)


def test_deaccumulate_running_mean_decreasing_rate() -> None:
    """Decreasing running-mean values (as radiation does through the evening) recover the step rate."""
    reset_frequency = pd.Timedelta.max
    lead_times = pd.to_timedelta(["0h", "1h", "2h", "3h"])

    # Step rates: step1=800, step2=600, step3=400.
    # Running means:
    #   A1 = (800) / 1                    = 800
    #   A2 = (800 + 600) / 2              = 700
    #   A3 = (800 + 600 + 400) / 3        = 600
    values = np.array([0.0, 800.0, 700.0, 600.0], dtype=np.float32)
    expected = np.array([np.nan, 800.0, 600.0, 400.0], dtype=np.float32)

    data_array = xr.DataArray(
        values,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "W m-2"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        accumulation_type="running_mean",
        invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
    )

    np.testing.assert_allclose(result.values, expected, equal_nan=True)


def test_deaccumulate_running_mean_step_size_transition() -> None:
    """ICON-EU switches from hourly to 3-hourly lead steps. Make sure both ranges recover."""
    reset_frequency = pd.Timedelta.max
    # Hourly through 3h then 3-hourly from 6h onwards (mimics the ICON-EU lead_time layout).
    lead_hours = [0, 1, 2, 3, 6, 9]
    lead_times = pd.to_timedelta(lead_hours, unit="h")

    # Step rates by the end of each window (uniform within each window):
    #   1h -> 300, 2h -> 500, 3h -> 700, (3-6h) -> 200, (6-9h) -> 100.
    # Running means over [0, t]:
    step_rates = {1: 300.0, 2: 500.0, 3: 700.0, 6: 200.0, 9: 100.0}
    # Cumulative sums of step_rate * step_duration give the integrated energy.
    cumulative_energy: list[float] = [0.0]
    for prev, curr in pairwise(lead_hours):
        cumulative_energy.append(
            cumulative_energy[-1] + step_rates[curr] * (curr - prev)
        )
    running_means = [
        e / h if h > 0 else 0.0
        for e, h in zip(cumulative_energy, lead_hours, strict=True)
    ]
    values = np.array(running_means, dtype=np.float32)

    expected = np.array(
        [np.nan, 300.0, 500.0, 700.0, 200.0, 100.0],
        dtype=np.float32,
    )

    data_array = xr.DataArray(
        values,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "W m-2"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        accumulation_type="running_mean",
        invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
    )

    np.testing.assert_allclose(result.values, expected, rtol=1e-5, equal_nan=True)


def test_deaccumulate_running_mean_nan_propagates() -> None:
    """Missing inputs keep the rate at that step (and the next dependent step) as NaN."""
    reset_frequency = pd.Timedelta.max
    lead_times = pd.to_timedelta(["0h", "1h", "2h", "3h"])

    values = np.array([0.0, np.nan, 600.0, 500.0], dtype=np.float32)
    # Step 1: prev A=0 (reset), sequence[1]=nan -> nan output.
    # Step 2: prev A=nan -> step_accumulation=nan -> output nan.
    # Step 3: prev A=600, sequence[3]=500 -> 500 + (500-600)*(2/1) = 300.
    expected = np.array([np.nan, np.nan, np.nan, 300.0], dtype=np.float32)

    data_array = xr.DataArray(
        values,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "W m-2"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        accumulation_type="running_mean",
        invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
    )

    np.testing.assert_allclose(result.values, expected, equal_nan=True)


def test_deaccumulate_running_mean_multidim() -> None:
    """Verify the running-mean branch works across parallelised leading dims."""
    reset_frequency = pd.Timedelta.max
    lead_times = pd.to_timedelta(["0h", "1h", "2h", "3h"])

    # Two ensemble members x a 1x1 spatial grid. Second member has every rate doubled.
    base = np.array([0.0, 800.0, 700.0, 600.0], dtype=np.float32)
    values = np.stack([base, 2 * base])[:, :, None, None]  # (member, lead_time, y, x)

    expected_member = np.array([np.nan, 800.0, 600.0, 400.0], dtype=np.float32)
    expected = np.stack([expected_member, 2 * expected_member])[:, :, None, None]

    data_array = xr.DataArray(
        values,
        coords={
            "ensemble_member": [0, 1],
            "lead_time": lead_times,
            "y": [0],
            "x": [0],
        },
        dims=["ensemble_member", "lead_time", "y", "x"],
        attrs={"units": "W m-2"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        accumulation_type="running_mean",
        invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
    )

    np.testing.assert_allclose(result.values, expected, rtol=1e-5, equal_nan=True)


def test_deaccumulate_running_mean_skip_step() -> None:
    """Skipped steps should be preserved and excluded from the calculation."""
    reset_frequency = pd.Timedelta.max
    lead_times = pd.to_timedelta(["0h", "1h", "2h", "3h"])

    values = np.array([0.0, 800.0, -999.0, 600.0], dtype=np.float32)
    skip = np.array([False, False, True, False])

    # Skip step 2. Step 3 sees previous_seconds=1h (tₜ₋₁ = 1h).
    # sequence[3] = 600 + (600 - 800) * (3600 / 7200) = 600 - 100 = 500.
    expected = np.array([np.nan, 800.0, -999.0, 500.0], dtype=np.float32)

    data_array = xr.DataArray(
        values,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "W m-2"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        skip_step=skip,
        accumulation_type="running_mean",
        invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
    )

    np.testing.assert_allclose(result.values, expected, equal_nan=True)


def test_deaccumulate_running_mean_invalid_below_threshold_raises() -> None:
    """A large drop in the running mean exceeds the threshold and raises."""
    reset_frequency = pd.Timedelta.max
    lead_times = pd.to_timedelta(["0h", "1h", "2h"])

    # A1=1000 (step rate 1000), A2=400 -> step rate = 400 + (400-1000)*1 = -200 W m-2.
    # -200 < RADIATION_INVALID_BELOW_THRESHOLD (-50), so expect a raise.
    values = np.array([0.0, 1000.0, 400.0], dtype=np.float32)

    data_array = xr.DataArray(
        values,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "W m-2"},
    )

    with pytest.raises(ValueError, match="below threshold"):
        deaccumulate_to_rates_inplace(
            data_array,
            dim="lead_time",
            reset_frequency=reset_frequency,
            accumulation_type="running_mean",
            invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
        )


def test_deaccumulate_running_mean_preserves_float32_precision_at_long_lead() -> None:
    """Pre-multiplying by cumulative seconds would lose float32 precision at long leads.

    The rearranged formula keeps every intermediate in the low thousands, so recovering
    constant-rate inputs should match to float32 roundoff even at 120h.
    """
    reset_frequency = pd.Timedelta.max
    hours = np.arange(0, 121, dtype=np.int64)
    lead_times = pd.to_timedelta(hours, unit="h")

    # Constant step rate of 400 W m-2 -> running mean is 400 for every non-zero step.
    values = np.where(hours == 0, 0.0, 400.0).astype(np.float32)
    expected = np.where(hours == 0, np.nan, 400.0).astype(np.float32)

    data_array = xr.DataArray(
        values,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "W m-2"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=reset_frequency,
        accumulation_type="running_mean",
        invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
    )

    np.testing.assert_allclose(result.values, expected, rtol=1e-6, equal_nan=True)


def test_deaccumulate_unknown_accumulation_type_raises() -> None:
    """The wrapper rejects an unknown accumulation_type before calling the numba kernel."""
    lead_times = pd.to_timedelta(["0h", "1h", "2h"])
    data_array = xr.DataArray(
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "W m-2"},
    )

    with pytest.raises(ValueError, match="Unknown accumulation_type"):
        deaccumulate_to_rates_inplace(
            data_array,
            dim="lead_time",
            reset_frequency=pd.Timedelta.max,
            accumulation_type="not-a-real-type",  # ty: ignore[invalid-argument-type]
        )


def _corrupt_spike_series() -> tuple[pd.TimedeltaIndex, np.ndarray]:
    """Accumulated precip (mm) with a single corrupt spike at 57h (mirrors 2024-09-05 m35).

    54h=43.79, 57h=65.25 (spike), 60h=36.62, 63h=41.63, 66h=46.01. The raw step rates are
    +7.15 (into the spike, not sign-flagged), -9.54 (out of it, flagged), then normal.
    """
    lead_times = pd.to_timedelta([0, 54, 57, 60, 63, 66], unit="h")
    accum = np.array([np.nan, 43.79, 65.25, 36.62, 41.63, 46.01], dtype=np.float32)
    return lead_times, accum


def test_deaccumulate_repair_none_is_default_and_leaves_nan() -> None:
    """Default (repair_implausible_drops='none') must NaN the implausible drop unchanged."""
    lead_times, accum = _corrupt_spike_series()
    data_array = xr.DataArray(
        accum,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )
    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=pd.Timedelta.max,
        expected_invalid_fraction=1.0,  # allow the NaN through without raising
    )
    # 60h step is the implausible drop -> NaN; the spurious +ve spike at 57h survives.
    assert np.isnan(result.values[3])
    assert result.values[2] > 0


def test_deaccumulate_repair_monotonic_removes_nan_and_spike() -> None:
    lead_times, accum = _corrupt_spike_series()
    raw_final = float(accum[-1])  # deaccumulate mutates accum in place
    data_array = xr.DataArray(
        accum,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )
    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=pd.Timedelta.max,
        repair_implausible_drops="monotonic",
    )
    rates = result.values
    assert np.isnan(rates[0])  # first step baseline is still NaN
    assert np.all(rates[1:] >= 0)  # non-decreasing accumulation -> no negative rates
    assert not np.isnan(rates[1:]).any()  # no spurious NaN remains
    # The reconstructed total (baseline 0 + integral of rates) stays close to the raw
    # final accumulation; isotonic repair adjusts endpoints slightly, so allow a few mm.
    dt = np.diff(lead_times.total_seconds().to_numpy())
    integrated = np.nansum(rates[1:] * dt)
    assert integrated == pytest.approx(raw_final, abs=2.0)


def test_deaccumulate_repair_temporal_removes_nan_and_spike() -> None:
    lead_times, accum = _corrupt_spike_series()
    data_array = xr.DataArray(
        accum,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )
    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=pd.Timedelta.max,
        repair_implausible_drops="temporal",
    )
    rates = result.values
    assert np.isnan(rates[0])
    assert not np.isnan(rates[1:]).any()  # negative step interpolated away
    assert np.all(rates[1:] >= 0)
    # Both corrupt steps (the +7.15 spike at 57h and the -9.54 drop at 60h) are replaced.
    assert rates[2] < 7.0 / 3600  # spike no longer present


def test_deaccumulate_repair_dip_series_monotonic() -> None:
    """A single low (dip) outlier, rather than a spike, is also repaired."""
    lead_times = pd.to_timedelta([0, 3, 6, 9], unit="h")
    # 3h=10mm, 6h=2mm (corrupt dip), 9h=12mm accumulated
    accum = np.array([np.nan, 10.0, 2.0, 12.0], dtype=np.float32)
    data_array = xr.DataArray(
        accum,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm s-1"},
    )
    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="lead_time",
        reset_frequency=pd.Timedelta.max,
        repair_implausible_drops="monotonic",
    )
    assert np.all(result.values[1:] >= 0)
    assert not np.isnan(result.values[1:]).any()


def test_deaccumulate_repair_rejects_running_mean() -> None:
    lead_times = pd.to_timedelta(["0h", "1h", "2h"], unit=None)
    data_array = xr.DataArray(
        np.array([0.0, 1000.0, 400.0], dtype=np.float32),
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "W m-2"},
    )
    with pytest.raises(NotImplementedError, match="accumulation_type='accumulated'"):
        deaccumulate_to_rates_inplace(
            data_array,
            dim="lead_time",
            reset_frequency=pd.Timedelta.max,
            accumulation_type="running_mean",
            repair_implausible_drops="monotonic",
        )


def test_deaccumulate_non_reset_aligned_first_step_nan() -> None:
    """Test when first step is NOT reset-aligned AND the first value is NaN.

    When the first value is NaN and it's not a reset point, subsequent values
    that depend on it (before the next reset) will also be NaN since we can't
    compute a valid rate without knowing the baseline accumulation.
    """
    reset_frequency = pd.Timedelta(hours=6)

    # Times starting at hour 23 (NOT a reset point)
    times = pd.DatetimeIndex(
        [
            "2024-01-01T23:00",  # 5h since reset, NOT a reset point, value is NaN
            "2024-01-02T00:00",  # 6h since reset, IS a reset point
            "2024-01-02T01:00",  # 1h since new reset at 00:00
        ]
    )

    # First value is NaN (missing data at hour 23)
    # Hour 0: 5.0 accumulated since 18:00
    # Hour 1: 1.0 accumulated since 00:00 reset
    accumulations = np.array([np.nan, 5.0, 1.0], dtype=np.float32)

    # Expected rates:
    # Hour 23: NaN (first step always NaN)
    # Hour 0: NaN (because previous_accumulation was NaN, so 5.0 - NaN = NaN)
    # Hour 1: 1.0 / 3600 (fresh window after reset, baseline is 0)
    expected_rates = np.array([np.nan, np.nan, 1.0 / 3600], dtype=np.float32)

    data_array = xr.DataArray(
        accumulations,
        coords={"time": times},
        dims=["time"],
        attrs={"units": "mm s-1"},
    )

    result = deaccumulate_to_rates_inplace(
        data_array,
        dim="time",
        reset_frequency=reset_frequency,
    )

    np.testing.assert_allclose(
        result.values,
        expected_rates,
        rtol=1e-6,
        equal_nan=True,
    )
