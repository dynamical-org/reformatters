from typing import Final

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.noaa.gefs.deaccumulation import (
    deaccumulate_to_rates_inplace,
)

SECONDS_PER_HOUR: Final[int] = 60 * 60


def test_deaccumulate_3hourly_from_06_12utc() -> None:
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


def test_deaccumulate_1d_case() -> None:
    mm_hr = SECONDS_PER_HOUR

    setup = [
        # 3 hourly step:
        {"lt": 0, "in": np.nan, "out": np.nan},
        {"lt": 3, "in": 4 * mm_hr * 3, "out": 4.0},
        {"lt": 6, "in": 4 * mm_hr * 3, "out": 0.0}, # 4mm accumulated in first 3 hours, no accumulation in next 3 hours  # fmt: off
        {"lt": 9, "in": 0 * mm_hr * 3, "out": 0.0},
        {"lt": 12, "in": 2 * mm_hr * 3, "out": 2.0}, # 0 mm accumulated in first 3 hours, 2 mm accumulated in next 3 hours
        # 6 hourly step:
        {"lt": 18, "in": 3 * mm_hr * 6, "out": 3.0},
        {"lt": 24, "in": 7 * mm_hr * 6, "out": 7.0},
    ]  # fmt: off

    lead_times = pd.to_timedelta([step["lt"] for step in setup], unit="h")
    data = np.array([step["in"] for step in setup])
    expected = np.array([step["out"] for step in setup])

    data_array = xr.DataArray(
        data,
        coords={"lead_time": lead_times},
        dims=["lead_time"],
        attrs={"units": "mm/s"},
    )

    result = deaccumulate_to_rates_inplace(data_array, dim="lead_time")

    np.testing.assert_equal(result.values, expected)
