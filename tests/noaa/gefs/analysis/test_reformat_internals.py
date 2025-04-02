from collections.abc import Sequence

import pandas as pd

from reformatters.noaa.gefs.analysis.reformat_internals import (
    filter_available_times,
    generate_chunk_coordinates,
)
from reformatters.noaa.gefs.read_data import EnsembleSourceFileCoords


def test_generate_chunk_coordinates_reforecast_with_hour_0() -> None:
    """Test generate_chunk_coordinates for reforecast period with hour 0 values."""
    # Use a specific date range around the reforecast boundary
    times = pd.date_range(start="2019-12-30T00:00", end="2020-01-02T00:00", freq="3h")
    ensemble_member = 0
    var_has_hour_0_values = True

    result = generate_chunk_coordinates(times, ensemble_member, var_has_hour_0_values)

    assert "ensemble" in result
    ensemble_coords: Sequence[EnsembleSourceFileCoords] = result["ensemble"]  # type: ignore[assignment]

    # Filter the times just as the function does
    filtered_times = filter_available_times(times)

    # Select some specific samples to verify directly
    # Sample 1: 2019-12-30T00:00 - This is a reforecast date with init_time = valid_time
    # But there are no hour 0 values in the reforecast period, so we shift back one forecast
    coords_dec30_00 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2019-12-30T00:00")
    )
    assert coords_dec30_00["init_time"] == pd.Timestamp("2019-12-29T00:00")
    assert coords_dec30_00["lead_time"] == pd.Timedelta("24 hours")
    assert coords_dec30_00["ensemble_member"] == 0

    # Sample 2: 2019-12-30T03:00 - This is 3 hours after an init time
    coords_dec30_03 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2019-12-30T03:00")
    )
    assert coords_dec30_03["init_time"] == pd.Timestamp("2019-12-30T00:00")
    assert coords_dec30_03["lead_time"] == pd.Timedelta("3 hours")
    assert coords_dec30_03["ensemble_member"] == 0

    # Sample 3: 2019-12-31T12:00 - Last day of reforecast period
    coords_dec31_12 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2019-12-31T12:00")
    )
    assert coords_dec31_12["init_time"] == pd.Timestamp("2019-12-31T00:00")
    assert coords_dec31_12["lead_time"] == pd.Timedelta("12 hours")
    assert coords_dec31_12["ensemble_member"] == 0

    # Sample 4: 2020-01-01T00:00 - First time after reforecast period
    coords_jan01_00 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-01-01T00:00")
    )
    assert coords_jan01_00["init_time"] == pd.Timestamp("2020-01-01T00:00")
    assert coords_jan01_00["lead_time"] == pd.Timedelta("0 hours")
    assert coords_jan01_00["ensemble_member"] == 0

    # Sample 5: 2020-01-01T06:00 - After reforecast period
    coords_jan01_06 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-01-01T06:00")
    )
    assert coords_jan01_06["init_time"] == pd.Timestamp("2020-01-01T06:00")
    assert coords_jan01_06["lead_time"] == pd.Timedelta("0 hours")
    assert coords_jan01_06["ensemble_member"] == 0

    # Sample 6: 2020-01-01T12:00 - After reforecast period
    coords_jan01_12 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-01-01T12:00")
    )
    assert coords_jan01_12["init_time"] == pd.Timestamp("2020-01-01T12:00")
    assert coords_jan01_12["lead_time"] == pd.Timedelta("0 hours")
    assert coords_jan01_12["ensemble_member"] == 0

    # Verify we have the right number of coordinates
    assert len(ensemble_coords) == len(filtered_times)


def test_generate_chunk_coordinates_reforecast_without_hour_0() -> None:
    """Test generate_chunk_coordinates for reforecast period without hour 0 values."""
    # Use a specific date range around the reforecast boundary
    times = pd.date_range(start="2019-12-30T00:00", end="2020-01-02T00:00", freq="3h")
    ensemble_member = 0
    var_has_hour_0_values = False

    result = generate_chunk_coordinates(times, ensemble_member, var_has_hour_0_values)

    assert "ensemble" in result
    ensemble_coords: Sequence[EnsembleSourceFileCoords] = result["ensemble"]  # type: ignore[assignment]

    # Filter the times just as the function does
    filtered_times = filter_available_times(times)

    # Sample 1: 2019-12-30T00:00 - This is a reforecast which does not contain hour 0 values
    coords_dec30_00 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2019-12-30T00:00")
    )
    assert coords_dec30_00["init_time"] == pd.Timestamp("2019-12-29T00:00")
    assert coords_dec30_00["lead_time"] == pd.Timedelta("24 hours")
    assert coords_dec30_00["ensemble_member"] == 0

    # Sample 2: 2019-12-30T03:00 - This is 3 hours after an init time
    coords_dec30_03 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2019-12-30T03:00")
    )
    assert coords_dec30_03["init_time"] == pd.Timestamp("2019-12-30T00:00")
    assert coords_dec30_03["lead_time"] == pd.Timedelta("3 hours")
    assert coords_dec30_03["ensemble_member"] == 0

    # Sample 3: 2020-01-01T00:00 - First time after reforecast period but needs to use
    # previous init time because of var_has_hour_0_values=False
    coords_jan01_00 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-01-01T00:00")
    )
    assert coords_jan01_00["init_time"] == pd.Timestamp("2019-12-31T00:00")
    assert coords_jan01_00["lead_time"] == pd.Timedelta("24 hours")
    assert coords_jan01_00["ensemble_member"] == 0

    # Sample 4: 2020-01-01T06:00 - After reforecast period
    coords_jan01_06 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-01-01T06:00")
    )
    assert coords_jan01_06["init_time"] == pd.Timestamp("2020-01-01T00:00")
    assert coords_jan01_06["lead_time"] == pd.Timedelta("6 hours")
    assert coords_jan01_06["ensemble_member"] == 0

    # Sample 5: 2020-01-01T12:00 - After reforecast period
    coords_jan01_12 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-01-01T12:00")
    )
    assert coords_jan01_12["init_time"] == pd.Timestamp("2020-01-01T06:00")
    assert coords_jan01_12["lead_time"] == pd.Timedelta("6 hours")
    assert coords_jan01_12["ensemble_member"] == 0

    # Verify we have the right number of coordinates
    assert len(ensemble_coords) == len(filtered_times)


def test_generate_chunk_coordinates_v12_transition_with_hour_0() -> None:
    """Test generate_chunk_coordinates for transition between pre- and post-GEFS v12 with hour 0 values."""
    # Time range around the GEFS v12 transition on 2020-10-01
    times = pd.date_range(start="2020-09-30T00:00", end="2020-10-02T00:00", freq="3h")
    ensemble_member = 0
    var_has_hour_0_values = True

    result = generate_chunk_coordinates(times, ensemble_member, var_has_hour_0_values)

    assert "ensemble" in result
    ensemble_coords: Sequence[EnsembleSourceFileCoords] = result["ensemble"]  # type: ignore[assignment]

    # Filter the times just as the function does
    filtered_times = filter_available_times(times)

    # Sample 1: 2020-09-30T00:00 - This is pre-GEFS v12 with hour 0
    coords_sep30_00 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-09-30T00:00")
    )
    assert coords_sep30_00["init_time"] == pd.Timestamp("2020-09-30T00:00")
    assert coords_sep30_00["lead_time"] == pd.Timedelta("0 hours")
    assert coords_sep30_00["ensemble_member"] == 0

    # Sample 2: 2020-09-30T06:00 - Still pre-GEFS v12
    coords_sep30_06 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-09-30T06:00")
    )
    assert coords_sep30_06["init_time"] == pd.Timestamp("2020-09-30T06:00")
    assert coords_sep30_06["lead_time"] == pd.Timedelta("0 hours")
    assert coords_sep30_06["ensemble_member"] == 0

    # Sample 3: 2020-10-01T00:00 - First time of GEFS v12 period
    coords_oct01_00 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-10-01T00:00")
    )
    assert coords_oct01_00["init_time"] == pd.Timestamp("2020-10-01T00:00")
    assert coords_oct01_00["lead_time"] == pd.Timedelta("0 hours")
    assert coords_oct01_00["ensemble_member"] == 0

    # Sample 4: 2020-10-01T03:00 - Post GEFS v12 with 3-hourly data
    coords_oct01_03 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-10-01T03:00")
    )
    assert coords_oct01_03["init_time"] == pd.Timestamp("2020-10-01T00:00")
    assert coords_oct01_03["lead_time"] == pd.Timedelta("3 hours")
    assert coords_oct01_03["ensemble_member"] == 0

    # Sample 5: 2020-10-01T06:00 - Post GEFS v12 with 6-hourly data
    coords_oct01_06 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-10-01T06:00")
    )
    assert coords_oct01_06["init_time"] == pd.Timestamp("2020-10-01T06:00")
    assert coords_oct01_06["lead_time"] == pd.Timedelta("0 hours")
    assert coords_oct01_06["ensemble_member"] == 0

    # Verify we have the right number of coordinates
    assert len(ensemble_coords) == len(filtered_times)


def test_generate_chunk_coordinates_v12_transition_without_hour_0() -> None:
    """Test generate_chunk_coordinates for transition between pre- and post-GEFS v12 without hour 0 values."""
    # Time range around the GEFS v12 transition on 2020-10-01
    times = pd.date_range(start="2020-09-30T00:00", end="2020-10-02T00:00", freq="3h")
    ensemble_member = 0
    var_has_hour_0_values = False

    result = generate_chunk_coordinates(times, ensemble_member, var_has_hour_0_values)

    assert "ensemble" in result
    ensemble_coords: Sequence[EnsembleSourceFileCoords] = result["ensemble"]  # type: ignore[assignment]

    # Filter the times just as the function does
    filtered_times = filter_available_times(times)

    # Sample 1: 2020-09-30T00:00 - This is pre-GEFS v12 on hour 0
    coords_sep30_00 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-09-30T00:00")
    )
    assert coords_sep30_00["init_time"] == pd.Timestamp("2020-09-29T18:00")
    assert coords_sep30_00["lead_time"] == pd.Timedelta("6 hours")
    assert coords_sep30_00["ensemble_member"] == 0

    # Sample 2: 2020-09-30T06:00 - Still pre-GEFS v12 on hour 6
    coords_sep30_06 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-09-30T06:00")
    )
    assert coords_sep30_06["init_time"] == pd.Timestamp("2020-09-30T00:00")
    assert coords_sep30_06["lead_time"] == pd.Timedelta("6 hours")
    assert coords_sep30_06["ensemble_member"] == 0

    # Sample A3: 2020-10-01T00:00 - First time of GEFS v12 period on hour 0
    coords_oct01_00 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-10-01T00:00")
    )
    assert coords_oct01_00["init_time"] == pd.Timestamp("2020-09-30T18:00")
    assert coords_oct01_00["lead_time"] == pd.Timedelta("6 hours")
    assert coords_oct01_00["ensemble_member"] == 0

    # Sample 4: 2020-10-01T03:00 - Post GEFS v12 with 3-hourly data
    coords_oct01_03 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-10-01T03:00")
    )
    assert coords_oct01_03["init_time"] == pd.Timestamp("2020-10-01T00:00")
    assert coords_oct01_03["lead_time"] == pd.Timedelta("3 hours")
    assert coords_oct01_03["ensemble_member"] == 0

    # Sample 5: 2020-10-01T06:00 - Post GEFS v12 where we use the previous init time
    coords_oct01_06 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-10-01T06:00")
    )
    assert coords_oct01_06["init_time"] == pd.Timestamp("2020-10-01T00:00")
    assert coords_oct01_06["lead_time"] == pd.Timedelta("6 hours")
    assert coords_oct01_06["ensemble_member"] == 0

    # Sample 6: 2020-10-01T09:00 - Post GEFS v12
    coords_oct01_09 = next(
        coord
        for coord in ensemble_coords
        if coord["lead_time"] + coord["init_time"] == pd.Timestamp("2020-10-01T09:00")
    )
    assert coords_oct01_09["init_time"] == pd.Timestamp("2020-10-01T06:00")
    assert coords_oct01_09["lead_time"] == pd.Timedelta("3 hours")
    assert coords_oct01_09["ensemble_member"] == 0

    # Verify we have the right number of coordinates
    assert len(ensemble_coords) == len(filtered_times)


def test_filter_available_times() -> None:
    start_time = pd.Timestamp("2019-12-31T00:00")
    end_time = pd.Timestamp("2020-01-01T21:00")

    # Create a date range with 3-hourly steps
    times = pd.date_range(start=start_time, end=end_time, freq="3h")

    filtered_times = filter_available_times(times)

    # Times from 2019-12-31 (before the new year) should be 3-hourly
    before_new_year = filtered_times[filtered_times < pd.Timestamp("2020-01-01")]
    expected_before = pd.DatetimeIndex(
        [
            pd.Timestamp("2019-12-31T00:00"),
            pd.Timestamp("2019-12-31T03:00"),
            pd.Timestamp("2019-12-31T06:00"),
            pd.Timestamp("2019-12-31T09:00"),
            pd.Timestamp("2019-12-31T12:00"),
            pd.Timestamp("2019-12-31T15:00"),
            pd.Timestamp("2019-12-31T18:00"),
            pd.Timestamp("2019-12-31T21:00"),
        ]
    )
    pd.testing.assert_index_equal(before_new_year, expected_before)

    # Times from 2020-01-01 (after the new year) should be 6-hourly
    after_new_year = filtered_times[filtered_times >= pd.Timestamp("2020-01-01")]
    expected_after = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-01T00:00"),
            pd.Timestamp("2020-01-01T06:00"),
            pd.Timestamp("2020-01-01T12:00"),
            pd.Timestamp("2020-01-01T18:00"),
        ]
    )
    pd.testing.assert_index_equal(after_new_year, expected_after)


def test_filter_available_times_current_archive_boundary() -> None:
    start_time = pd.Timestamp("2020-09-30T00:00")
    end_time = pd.Timestamp("2020-10-01T18:00")

    # Create a date range with 3-hourly steps
    times = pd.date_range(start=start_time, end=end_time, freq="3h")

    filtered_times = filter_available_times(times)

    # Times before Oct 1 should be 6-hourly
    before_oct = filtered_times[filtered_times < pd.Timestamp("2020-10-01")]
    expected_before = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-09-30T00:00"),
            pd.Timestamp("2020-09-30T06:00"),
            pd.Timestamp("2020-09-30T12:00"),
            pd.Timestamp("2020-09-30T18:00"),
        ]
    )
    pd.testing.assert_index_equal(before_oct, expected_before)

    # Times from Oct 1 onwards should be 3-hourly
    after_oct = filtered_times[filtered_times >= pd.Timestamp("2020-10-01")]
    expected_after = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-10-01T00:00"),
            pd.Timestamp("2020-10-01T03:00"),
            pd.Timestamp("2020-10-01T06:00"),
            pd.Timestamp("2020-10-01T09:00"),
            pd.Timestamp("2020-10-01T12:00"),
            pd.Timestamp("2020-10-01T15:00"),
            pd.Timestamp("2020-10-01T18:00"),
        ]
    )
    pd.testing.assert_index_equal(after_oct, expected_after)
