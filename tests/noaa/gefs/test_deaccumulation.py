from typing import Final

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.gefs.deaccumulation import (
    deaccumulate_to_rates_inplace,
)

SECONDS_PER_HOUR: Final[int] = 60 * 60


def test_deaccumulate_higher_dimensional() -> None:
    # Create test data
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
    data = np.expand_dims(data, axis=(3, 4))

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
    expected = np.expand_dims(expected, axis=(3, 4))

    result = deaccumulate_to_rates_inplace(data_array, dim="lead_time")

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_1d_3_and_6_hour_normal_cases() -> None:
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

    result = deaccumulate_to_rates_inplace(data_array, dim="lead_time")

    np.testing.assert_equal(result.values, expected)


def test_deaccumulate_1d_3_and_6_hour_small_accumulation_decreases() -> None:
    values = [
        # 3 hourly step:
        {"lt": 0, "in": 0., "out": 0.},
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
        attrs={"units": "mm/s"},
    )

    with pytest.raises(ValueError) as e:
        result = deaccumulate_to_rates_inplace(data_array, dim="lead_time")

        np.testing.assert_equal(result.values, expected)

    assert e.value.args[0] == "Over 5% (1 total) values were clamped to 0"


def test_deaccumulate_1d_3_and_6_hour_large_accumulation_decreases() -> None:
    values = [
        {"lt": 0, "in": np.nan, "out": np.nan},
        {"lt": 3, "in": 2, "out": 2.0 / (3 * SECONDS_PER_HOUR)},
        {"lt": 6, "in": 1.8, "out": np.nan},  # negative accumulation too large, set to NaN
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

    with pytest.raises(ValueError) as e:
        result = deaccumulate_to_rates_inplace(data_array, dim="lead_time")

        np.testing.assert_equal(result.values, expected)

    assert e.value.args[0] == "Found 1 values below threshold"
