from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import ceil

import numpy as np

VIRTUAL_POLL_LEAD = 15 * 60
VIRTUAL_SHORT_POLL_LEAD = 3 * 60


@dataclass(frozen=True)
class ArrivalStats:
    count: int
    p50: float
    p95: float
    p99: float
    earliest: float


def arrival_stats(values: Sequence[float]) -> ArrivalStats:
    assert values, "At least one arrival is required"
    assert all(np.isfinite(value) for value in values)
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return ArrivalStats(len(values), float(p50), float(p95), float(p99), min(values))


def schedule_offset(stats: ArrivalStats) -> float:
    return ceil_minute(stats.p99 + 3 * 60)


def max_normal_gap(values: Sequence[float]) -> float:
    p50, p90 = np.percentile(values, [50, 90])
    return max(3 * 60, float(p90 - p50))


def max_normal_cut(values: Sequence[float]) -> float:
    assert values, "At least one arrival is required"
    ordered = sorted(values)
    median = float(np.median(ordered))
    gap = max_normal_gap(values)
    return next(
        (
            left
            for left, right in pairwise(ordered)
            if left >= median and right - left > gap
        ),
        ordered[-1],
    )


def regime_start(
    samples: Sequence[tuple[float, float]], end: float
) -> tuple[float | None, bool]:
    boundary = None
    while True:
        restart, suspected = _next_regime_start(samples, end)
        if restart is None or restart == boundary:
            return boundary, suspected
        boundary = restart
        samples = [sample for sample in samples if sample[0] >= boundary]


def _next_regime_start(
    samples: Sequence[tuple[float, float]], end: float
) -> tuple[float | None, bool]:
    ordered = sorted(samples)
    if len(ordered) < 10:
        return None, False
    values = np.array([value for _, value in ordered])

    def loss(index: int) -> float:
        left, right = values[:index], values[index:]
        return float(
            np.abs(left - np.median(left)).sum()
            + np.abs(right - np.median(right)).sum()
        )

    boundary = min(range(5, len(ordered) - 4), key=loss)
    if (
        abs(float(np.median(values[:boundary]) - np.median(values[boundary:])))
        < 15 * 60
    ):
        split = next(
            (i for i, (anchor, _) in enumerate(ordered) if anchor >= end - 14 * 86400),
            len(ordered),
        )
        if split < 5:
            return None, False
        cut = max_normal_cut(values[:split].tolist())
        boundary = len(ordered)
        while boundary > split and values[boundary - 1] > cut:
            boundary -= 1
        if (
            len(ordered) - boundary < 2
            or (len(ordered) - boundary) / len(ordered) < 0.01
        ):
            return None, False
    if ordered[-1][0] - ordered[boundary][0] <= 86400:
        return None, True
    return ordered[boundary][0], False


def tail_dominated(values: Sequence[float]) -> bool:
    cut = max_normal_cut(values)
    return (
        schedule_offset(arrival_stats(values)) - ceil_minute(cut + 3 * 60) > 15 * 60
        or sum(value > cut for value in values) / len(values) > 0.01
    )


def read_rates(
    arrivals: Sequence[float | None],
    fire: float,
    startup: float | None,
    runtime: float | None,
    validation: float | None,
) -> tuple[float | None, float | None]:
    if not arrivals or startup is None:
        return None, None

    def incomplete(offset: float) -> float:
        return sum(value is None or value > offset for value in arrivals) / len(
            arrivals
        )

    repass = incomplete(fire + startup)
    page = (
        incomplete(fire + 2 * startup + runtime + validation + 5 * 60)
        if runtime is not None and validation is not None
        else None
    )
    return repass, page


def virtual_fire(
    earliest: float, startup: float, lead: float = VIRTUAL_POLL_LEAD
) -> float:
    return max(60, earliest - startup - lead)


def virtual_poll_lead(
    earliest: float, startup: float, interval: float, completion_p99: float | None
) -> float:
    fire = (virtual_fire(earliest, startup) // 60) * 60
    cutoff = virtual_poll_cutoff(fire, interval, startup)
    return (
        VIRTUAL_SHORT_POLL_LEAD
        if completion_p99 is not None and cutoff < completion_p99
        else VIRTUAL_POLL_LEAD
    )


def virtual_poll_cutoff(fire: float, interval: float, startup: float) -> float:
    return fire + interval + startup + 60


def virtual_deadline(interval: float, startup: float, validation_max: float) -> float:
    return interval + startup + 60 + validation_max + 30


def currency_delay(
    offset: float,
    anchor_minus_label: float,
    interval: float,
    anchor_spacing: float,
    *,
    virtual: bool,
) -> float:
    if virtual:
        return offset + 4 * 60 + anchor_minus_label
    return offset + anchor_minus_label + max(0, interval - anchor_spacing)


def ceil_minute(seconds: float) -> float:
    return ceil(seconds / 60) * 60
