from collections.abc import Sequence
from dataclasses import dataclass
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
    return {
        "draft": stats.p95 + max(stats.p95 - stats.p50, 5 * 60),
        "p99+1m": stats.p99 + 60,
        "spread, min 3m": stats.p95 + max(stats.p95 - stats.p50, 3 * 60),
    }


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
