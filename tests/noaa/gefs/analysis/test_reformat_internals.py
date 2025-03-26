import pandas as pd

from reformatters.noaa.gefs.analysis.reformat_internals import filter_available_times


def test_filter_available_times() -> None:
    start_time = pd.Timestamp("2019-12-31T00:00")
    end_time = pd.Timestamp("2020-01-01T21:00")

    # Create a date range with 3-hourly steps
    times = pd.date_range(start=start_time, end=end_time, freq="3H")

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
    times = pd.date_range(start=start_time, end=end_time, freq="3H")

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
