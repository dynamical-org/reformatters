from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import ceil

import numpy as np


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


def late_threshold(stats: ArrivalStats) -> float:
    return stats.p95 + max(stats.p95 - stats.p50, 15 * 60)


def schedule_candidates(stats: ArrivalStats) -> dict[str, float]:
    candidates = {
        "draft": stats.p95 + max(stats.p95 - stats.p50, 5 * 60),
        "p99+1m": stats.p99 + 60,
        "spread, min 3m": stats.p95 + max(stats.p95 - stats.p50, 3 * 60),
    }
    candidates["chosen"] = schedule_offset(stats)
    return candidates


def schedule_offset(stats: ArrivalStats) -> float:
    return min(stats.p95 + max(stats.p95 - stats.p50, 5 * 60), stats.p99 + 60)


def max_normal_cut(values: Sequence[float]) -> float:
    assert values, "At least one arrival is required"
    ordered = sorted(values)
    median = float(np.median(ordered))
    gaps = [
        (right - left, left)
        for left, right in pairwise(ordered)
        if left >= median and right > left
    ]
    return max(gaps, key=lambda pair: pair[0])[1] if gaps else ordered[-1]


def recent_shift(
    samples: Sequence[tuple[float, float]], end: float
) -> tuple[float, float] | None:
    recent = [value for anchor, value in samples if end - 7 * 86400 <= anchor < end]
    prior = [
        value
        for anchor, value in samples
        if end - 35 * 86400 <= anchor < end - 7 * 86400
    ]
    if len(recent) < 5 or len(prior) < 5:
        return None
    before, after = float(np.median(prior)), float(np.median(recent))
    return (before, after) if abs(after - before) >= 15 * 60 else None


def candidate_outcomes(
    arrivals: Sequence[float],
    offset: float,
    startup: float | None,
    runtime: float | None,
) -> tuple[float, float | None]:
    fraction = sum(value > offset for value in arrivals) / len(arrivals)
    on_time = [offset - value for value in arrivals if value <= offset]
    latency = (
        float(np.median(on_time)) + startup + runtime
        if on_time and startup is not None and runtime is not None
        else None
    )
    return fraction, latency


def virtual_fire(earliest: float, startup: float) -> float:
    return earliest - startup - 60


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
