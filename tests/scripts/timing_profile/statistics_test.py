import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.timing_profile.statistics import (
    arrival_stats,
    ceil_minute,
    currency_delay,
    max_normal_cut,
    max_normal_gap,
    read_rates,
    regime_start,
    schedule_offset,
    tail_dominated,
    virtual_deadline,
    virtual_fire,
)


def test_linear_percentiles_and_earliest_include_recurring_fast_arrivals() -> None:
    stats = arrival_stats([60, 60, 120, 180, 240])
    assert stats.count == 5
    assert stats.p50 == 120
    assert stats.p95 == pytest.approx(228)
    assert stats.p99 == pytest.approx(237.6)
    assert stats.earliest == 60


@pytest.mark.parametrize(
    ("values", "expected"), [([600, 600, 600], 780), ([0, 3600], 3780)]
)
def test_schedule_and_backstop_use_rounded_p99_plus_three_minutes(
    values: list[float], expected: float
) -> None:
    assert schedule_offset(arrival_stats(values)) == expected


def test_read_model_counts_incomplete_at_start_and_first_repass_read() -> None:
    # Reads are at 310 and 670 seconds; availability at a read is sufficient.
    assert read_rates([60, 310, 311, 670, 671, None], 300, 10, 30, 20) == (4 / 6, 2 / 6)
    assert read_rates([600], 300, None, 30, 20) == (None, None)
    assert read_rates([600], 300, 10, None, 20) == (1, None)
    assert read_rates([600], 300, 10, 30, None) == (1, None)
    assert read_rates([], 300, 10, 30, 20) == (None, None)


def test_virtual_fire_and_deadline_use_startup_and_maximum_validation() -> None:
    assert virtual_fire(3000, 40) == 2060
    assert virtual_fire(900, 40) == 60
    assert virtual_fire(-60, 40) == 60
    assert virtual_deadline(3600, 40, 120) == 3850


def test_currency_delay_accounts_for_labels_and_multiple_anchors() -> None:
    assert currency_delay(3000, 0, 10800, 3600, virtual=False) == 10200
    assert currency_delay(3600, 86400, 86400, 86400, virtual=False) == 90000
    assert currency_delay(3000, 1800, 3600, 1800, virtual=True) == 5040
    assert ceil_minute(3000.1) == 3060


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([600], 600),
        ([600, 600, 600], 600),
        ([0, 1000, 1001, 1002, 1003], 1003),
        ([1000] * 30 + [1001, 1400, 9000, 30000], 1001),
    ],
)
def test_max_normal_cut_uses_first_gap_above_median(
    values: list[float], expected: float
) -> None:
    assert max_normal_cut(values) == expected


def test_max_normal_gap_scales_with_the_upper_body() -> None:
    assert max_normal_gap([600] * 10) == 180
    assert max_normal_gap(list(range(0, 6000, 60))) > 180


@pytest.mark.parametrize(("before", "after"), [(600, 1800), (1800, 600)])
def test_median_shift_restarts_at_the_break_in_both_directions(
    before: float, after: float
) -> None:
    samples = [(day * 86400, before if day < 172 else after) for day in range(180)]
    assert regime_start(samples, 180 * 86400) == (172 * 86400, False)


def test_trailing_cluster_needs_more_than_one_day_of_persistence() -> None:
    samples = [(day * 86400, 1800 if day >= 178 else 600) for day in range(180)]
    assert regime_start(samples, 180 * 86400) == (None, True)
    assert regime_start([*samples[:-1], (179 * 86400, 600)], 180 * 86400) == (
        None,
        False,
    )
    assert regime_start(samples[-4:], 180 * 86400) == (None, False)


@pytest.mark.parametrize(
    ("days_back", "before", "after"),
    [(30, 600, 2400), (120, 2400, 600), (170, 2400, 600)],
)
def test_whole_history_split_keeps_the_entire_new_regime(
    days_back: int, before: float, after: float
) -> None:
    boundary = (180 - days_back) * 4
    samples = [
        (index * 21600, before if index < boundary else after) for index in range(720)
    ]
    assert regime_start(samples, 180 * 86400) == (boundary * 21600, False)


@pytest.mark.parametrize("usable_days", [77, 100])
def test_one_slow_day_with_partial_history_stays_suspected(usable_days: int) -> None:
    samples = [
        (index * 21600, 7200 if index >= 716 else 3600)
        for index in range((180 - usable_days) * 4, 720)
    ]
    assert regime_start(samples, 180 * 86400) == (None, True)


def test_scattered_outliers_do_not_restart_history() -> None:
    samples = [(day * 86400, 1800 if day in {174, 177} else 600) for day in range(180)]
    assert regime_start(samples, 180 * 86400) == (None, False)


@pytest.mark.parametrize("outliers", [1, 2])
def test_tail_dominated_checks_spread_and_outlier_mass(outliers: int) -> None:
    assert tail_dominated([600] * (100 - outliers) + [9000] * outliers) == (
        outliers > 1
    )
    assert tail_dominated([600] * 98 + [3000, 9000])
    assert tail_dominated([600] * 99 + [900000])
    assert not tail_dominated([600] * 100)


@pytest.mark.parametrize("days_after_last", [1, 30])
def test_real_aifs_step_keeps_the_same_break_as_history_ages(
    days_after_last: int,
) -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/aifs_ens_arrivals.json").read_text()
    )
    anchors = pd.date_range(
        fixture["start"],
        periods=len(fixture["arrivals_s"]),
        freq=pd.Timedelta(hours=fixture["frequency_hours"]),
    )
    samples = [
        (anchor.timestamp(), value)
        for anchor, value in zip(anchors, fixture["arrivals_s"], strict=True)
    ]
    end = (anchors[-1] + pd.Timedelta(days=days_after_last)).timestamp()
    boundary, suspected = regime_start(samples, end)
    assert not suspected
    assert boundary is not None
    assert boundary == pd.Timestamp("2026-10-20T06:00Z").timestamp()
    assert sum(anchor >= boundary for anchor, _ in samples) == 31
