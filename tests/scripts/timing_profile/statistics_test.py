import pytest

from scripts.timing_profile.statistics import (
    arrival_stats,
    candidate_outcomes,
    ceil_minute,
    currency_delay,
    late_threshold,
    schedule_candidates,
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
    ("values", "expected"), [([600, 600, 600], 1500), ([0, 3600], 5040)]
)
def test_late_threshold_floor_and_tail(values: list[float], expected: float) -> None:
    assert late_threshold(arrival_stats(values)) == expected


def test_all_schedule_candidates_have_independent_minimum_allowances() -> None:
    assert schedule_candidates(arrival_stats([600, 600, 600])) == {
        "draft": 900,
        "p99+1m": 660,
        "spread, min 3m": 780,
        "chosen": 660,
    }
    candidates = schedule_candidates(arrival_stats([0, 3600]))
    assert candidates["draft"] == 5040
    assert candidates["spread, min 3m"] == 5040
    assert candidates["p99+1m"] == 3624
    assert candidates["chosen"] == 3624


def test_candidate_outcomes_exclude_late_units_from_latency_estimate() -> None:
    fraction, latency = candidate_outcomes([60, 120, 240, 600], 300, 10, 20)
    assert fraction == 0.25
    assert latency == 210
    assert candidate_outcomes([60], 300, 10, None) == (0, None)
    assert candidate_outcomes([600], 300, 10, 20) == (1, None)


def test_virtual_fire_and_deadline_use_startup_and_maximum_validation() -> None:
    assert virtual_fire(3000, 40) == 2900
    assert virtual_deadline(3600, 40, 120) == 3850


def test_currency_delay_accounts_for_labels_and_multiple_anchors() -> None:
    assert currency_delay(3000, 0, 10800, 3600, virtual=False) == 10200
    assert currency_delay(3600, 86400, 86400, 86400, virtual=False) == 90000
    assert currency_delay(3000, 1800, 3600, 1800, virtual=True) == 5040
    assert ceil_minute(3000.1) == 3060
