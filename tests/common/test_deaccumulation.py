from typing import Final

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.deaccumulation import (
    PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
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
        with pytest.raises(ValueError, match="Found 1 values below threshold"):
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
