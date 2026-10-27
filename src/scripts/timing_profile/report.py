from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

import pandas as pd

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import ReformatCronJob
from reformatters.common.validation import CheckCurrentData

from .datasets import Unit, anchor_spacing, is_virtual
from .runtime import RuntimeStats
from .sources import Listing, listing_prefix
from .statistics import (
    arrival_stats,
    candidate_outcomes,
    ceil_minute,
    currency_delay,
    late_threshold,
    schedule_candidates,
    virtual_deadline,
    virtual_fire,
)


@dataclass(frozen=True)
class MeasuredUnit:
    unit: Unit
    first: float | None
    complete: float | None
    missing_files: int
    components: dict[str, float]


def aggregate_unit(
    unit: Unit, listings: dict[str, Listing], *, as_of: pd.Timestamp | None = None
) -> MeasuredUnit:
    available = []
    missing = 0
    components: dict[str, float] = {}
    for file in unit.files:
        timestamps = [listings[listing_prefix(url)].objects.get(url) for url in file]
        timestamps = [
            value if as_of is None or value is None or value <= as_of else None
            for value in timestamps
        ]
        if any(value is None for value in timestamps):
            missing += 1
            continue
        usable = max(value for value in timestamps if value is not None)
        offset = (usable - unit.anchor).total_seconds()
        available.append(offset)
        group = "0p25" if any(".0p25." in url for url in file) else "other"
        components[group] = max(components.get(group, float("-inf")), offset)
    return MeasuredUnit(
        unit,
        min(available) if available else None,
        max(available) if available and missing == 0 else None,
        missing,
        components,
    )


def duration(value: float | None) -> str:
    if value is None:
        return "unmeasured"
    sign = "-" if value < 0 else ""
    seconds = round(abs(value))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{sign}{hours:02}:{minutes:02}:{seconds:02}"


def cron_interval(
    cron: ReformatCronJob, end: pd.Timestamp, *, maximum: bool = False
) -> float:
    fire = cron.previous_fire_time(end)
    intervals = []
    for _ in range(40):
        previous = cron.previous_fire_time(fire - pd.Timedelta(seconds=1))
        intervals.append((fire - previous).total_seconds())
        fire = previous
    return max(intervals) if maximum else min(intervals)


def recommended_cron(offset: float, interval: float, cron: ReformatCronJob) -> str:
    rounded_minutes = round(offset / 60)
    minute = rounded_minutes % 60
    hour_shift = rounded_minutes // 60
    if interval < 3600:
        return f"{minute % round(interval / 60)}-59/{round(interval / 60)} * * * *"
    hours = sorted(
        {
            (hour + hour_shift) % 24
            for hour in range(0, 24, min(24, round(interval / 3600)))
        }
    )
    return f"{minute} {','.join(map(str, hours)) if len(hours) < 24 else '*'} {' '.join(cron.schedule.split()[2:])}"


def cycle_classes(
    units: Sequence[MeasuredUnit],
) -> list[tuple[str, list[MeasuredUnit]]]:
    by_hour: dict[int, list[MeasuredUnit]] = defaultdict(list)
    for unit in units:
        by_hour[unit.unit.anchor.hour].append(unit)
    measured = {
        hour: arrival_stats([u.complete for u in values if u.complete is not None]).p50
        for hour, values in by_hour.items()
        if any(u.complete is not None for u in values)
    }
    classes: list[list[int]] = []
    for hour in sorted(measured, key=lambda key: measured[key]):
        if not classes or measured[hour] - measured[classes[-1][0]] >= 15 * 60:
            classes.append([])
        classes[-1].append(hour)
    unmeasured = [hour for hour in by_hour if hour not in measured]
    if unmeasured:
        classes.append(unmeasured)
    return [
        (
            "/".join(f"{hour:02}" for hour in sorted(hours)) + "Z",
            [unit for hour in hours for unit in by_hour[hour]],
        )
        for hours in classes
    ]


def render_dataset(
    dataset: DynamicalDataset[Any, Any],
    cron: ReformatCronJob,
    units: Sequence[MeasuredUnit],
    runtime: RuntimeStats,
    end: pd.Timestamp,
    gap: str | None,
    source_notes: set[str],
) -> str:
    interval = cron_interval(cron, end)
    currency_interval = cron_interval(cron, end, maximum=True)
    virtual = is_virtual(dataset)
    current_delay = next(
        (
            validator.max_delay.total_seconds()
            for validator in dataset.validators()
            if isinstance(validator, CheckCurrentData)
        ),
        None,
    )
    lines = [
        f"## {dataset.dataset_id}",
        "",
        f"Current cron: `{cron.schedule}`; suspended: `{cron.suspend}`; max_delay: {duration(current_delay)}; pod deadline: {duration(cron.pod_active_deadline.total_seconds())}; Job deadline: {duration(cron.active_deadline.total_seconds()) if cron.active_deadline else 'unset'}. Fire interval min/max: {duration(interval)} / {duration(currency_interval)} (calendar-varying schedules use the minimum for deadlines, maximum for scalar currency).",
        "",
        f"Runtime: L estimate p95 {duration(runtime.startup_estimate)}; R p95 {duration(runtime.processing)}; V p95/max {duration(runtime.validation_p95)} / {duration(runtime.validation_max)}. Scheduled-ok n={runtime.scheduled_count}; matched inline validation n={runtime.validation_count}; non-ok update n={runtime.non_ok_count}; off-schedule ok n={runtime.off_schedule_count}. Check-ins: {runtime.first} .. {runtime.last} UTC.",
        "",
    ]
    if runtime.by_fire_hour:
        lines.extend(
            [
                "| Scheduled fire hour (UTC) | L estimate p95 | R p95 | V p95 / max | Scheduled-ok / matched validation n |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {hour:02} | {duration(values.startup_estimate)} | {duration(values.processing)} | {duration(values.validation_p95)} / {duration(values.validation_max)} | {values.scheduled_count} / {values.validation_count} |"
                    for hour, values in sorted(runtime.by_fire_hour.items())
                ],
                "",
            ]
        )
    if gap:
        lines.extend([f"Source: **not measured; keep constants** — {gap}.", ""])
        return "\n".join(lines)
    identities = sorted(
        {
            listing_prefix(url)
            for unit in units
            for file in unit.unit.files
            for url in file
        }
    )
    buckets = sorted(
        {url.split("/", 3)[0] + "//" + url.split("/", 3)[2] for url in identities}
    )
    lines.extend(
        [
            f"Source identity: {', '.join(buckets)}. {'; '.join(sorted(source_notes))}.",
            "",
        ]
    )
    tiers: dict[str, list[MeasuredUnit]] = defaultdict(list)
    for unit in units:
        tiers[unit.unit.tier].append(unit)
    if not tiers:
        lines.extend(
            [
                "Source: **not measured; keep constants** — no readiness units generated.",
                "",
            ]
        )
        return "\n".join(lines)
    recommendations: list[tuple[str, str, float | None, float, float]] = []
    for tier, tier_units in sorted(tiers.items()):
        lines.extend(
            [
                f"### {tier}",
                "",
                "| Cycle class | Complete / expected | Missing old / recent censored | First-file n / p50 / p95 / p99 / E | Unit p50 / p95 / p99 / T | Observed anchors (UTC) |",
                "|---|---:|---:|---|---|---|",
            ]
        )
        candidates = []
        for cycle_class, grouped in cycle_classes(tier_units):
            complete = [unit.complete for unit in grouped if unit.complete is not None]
            first = [unit.first for unit in grouped if unit.first is not None]
            horizon = max(complete) if complete else 0
            censored = sum(
                unit.complete is None
                and (end - unit.unit.anchor).total_seconds() <= horizon
                for unit in grouped
            )
            missing = len(grouped) - len(complete) - censored
            first_stats = arrival_stats(first) if first else None
            stats = arrival_stats(complete) if complete else None
            first_cell = (
                f"{len(first)} / "
                + " / ".join(
                    duration(value)
                    for value in (
                        first_stats.p50,
                        first_stats.p95,
                        first_stats.p99,
                        first_stats.earliest,
                    )
                )
                if first_stats
                else "not measured"
            )
            unit_cell = (
                " / ".join(
                    duration(value)
                    for value in (
                        stats.p50,
                        stats.p95,
                        stats.p99,
                        late_threshold(stats),
                    )
                )
                if stats
                else "not measured; keep constants"
            )
            observed = [unit.unit.anchor for unit in grouped if unit.first is not None]
            window = f"{min(observed)} .. {max(observed)}" if observed else "unmeasured"
            lines.append(
                f"| {cycle_class} | {len(complete)} / {len(grouped)} | {missing} / {censored} | {first_cell} | {unit_cell} | {window} |"
            )
            if stats and first_stats:
                threshold = late_threshold(first_stats if virtual else stats)
                for name, offset in schedule_candidates(stats).items():
                    fraction, latency = candidate_outcomes(
                        complete, offset, runtime.startup_estimate, runtime.processing
                    )
                    candidates.append(
                        f"| {cycle_class} | {name} | {duration(offset)} | {fraction:.2%} | {duration(latency)} |"
                    )
                if virtual:
                    offset = (
                        virtual_fire(first_stats.earliest, runtime.startup_estimate)
                        if runtime.startup_estimate is not None
                        else None
                    )
                    delay = currency_delay(
                        threshold,
                        grouped[0].unit.anchor_minus_label,
                        currency_interval,
                        anchor_spacing(dataset).total_seconds(),
                        virtual=True,
                    )
                else:
                    offset = schedule_candidates(stats)["draft"]
                    delay = currency_delay(
                        ceil_minute(offset),
                        grouped[0].unit.anchor_minus_label,
                        currency_interval,
                        anchor_spacing(dataset).total_seconds(),
                        virtual=False,
                    )
                recommendations.append(
                    (tier, cycle_class, offset, delay, late_threshold(stats))
                )
        lines.extend(
            [
                "",
                "| Cycle class | S candidate | S | Fraction of units arriving after S | Median on-time latency estimate (S - arrival + L + R) |",
                "|---|---|---:|---:|---:|",
                *candidates,
                "",
            ]
        )
    lines.extend(
        [
            "### Advisory constants",
            "",
            "| Tier / class | Per-class offset | max_delay | Backstop T (unit) |",
            "|---|---:|---:|---:|",
        ]
    )
    for tier, cycle_class, offset, delay, threshold in recommendations:
        lines.append(
            f"| {tier} / {cycle_class} | {duration(offset)} | {duration(ceil_minute(delay))} | {duration(threshold)} |"
        )
    lines.extend(
        render_single_cron(
            dataset, cron, tiers, recommendations, runtime, interval, units
        )
    )
    return "\n".join(lines)


def render_single_cron(
    dataset: DynamicalDataset[Any, Any],
    cron: ReformatCronJob,
    tiers: dict[str, list[MeasuredUnit]],
    recommendations: list[tuple[str, str, float | None, float, float]],
    runtime: RuntimeStats,
    interval: float,
    units: Sequence[MeasuredUnit],
) -> list[str]:
    lines: list[str] = []
    virtual = is_virtual(dataset)
    first_tier = min(tiers)
    offsets = [
        offset
        for tier, _, offset, _, _ in recommendations
        if tier == first_tier and offset is not None
    ]
    if offsets and dataset.dataset_id != "noaa-gefs-forecast-35-day-0-5-degree-virtual":
        chosen = min(offsets) if virtual else max(offsets)
        chosen = (chosen // 60) * 60 if virtual else ceil_minute(chosen)
        deadline = (
            virtual_deadline(interval, runtime.startup_estimate, runtime.validation_max)
            if virtual
            and runtime.startup_estimate is not None
            and runtime.validation_max is not None
            else (None if virtual else interval)
        )
        scalar_delay = max(
            delay for tier, _, _, delay, _ in recommendations if tier == first_tier
        )
        scalar_delay = ceil_minute(scalar_delay)
        current_delay = next(
            (
                validator.max_delay.total_seconds()
                for validator in dataset.validators()
                if isinstance(validator, CheckCurrentData)
            ),
            None,
        )
        new_cron = recommended_cron(chosen, interval, cron)
        lines.extend(
            [
                "",
                "| Constant | Current | Recommended | Recommended minus current |",
                "|---|---|---|---:|",
                f"| Cron | `{cron.schedule}` | `{new_cron}` | anchor attribution not inferred |",
                f"| max_delay | {duration(current_delay)} | {duration(scalar_delay)} | {duration(scalar_delay - current_delay) if current_delay is not None else 'unmeasured'} |",
                f"| Pod deadline | {duration(cron.pod_active_deadline.total_seconds())} | {duration(ceil(deadline)) if deadline is not None else 'unmeasured; keep constants'} | {duration(ceil(deadline) - cron.pod_active_deadline.total_seconds()) if deadline is not None else 'unmeasured'} |",
                f"| Job deadline | {duration(cron.active_deadline.total_seconds()) if cron.active_deadline else 'unset'} | {duration(ceil(deadline)) if deadline is not None else 'unmeasured; keep constants'} | {duration(ceil(deadline) - cron.active_deadline.total_seconds()) if deadline is not None and cron.active_deadline else 'unset or unmeasured'} |",
            ]
        )
        lines.extend(
            [
                "",
                f"One-cron offset: {duration(chosen)}. {'Uses earliest class and first-file E.' if virtual else 'Uses latest class and draft S; selection remains manual.'}",
                "",
            ]
        )
        if virtual:
            lines.extend(
                [
                    f"Validation grace: {duration(ceil(runtime.validation_max + 30)) if runtime.validation_max is not None else 'unmeasured; keep constants'} (maximum observed V plus allowance).",
                    "",
                ]
            )
    if dataset.dataset_id == "noaa-gefs-forecast-35-day-0-5-degree-virtual":
        lines.extend(
            [
                "",
                "One-cron recommendation: keep named-exception window constants; review the measured tier arrivals manually.",
                "",
            ]
        )
    if dataset.dataset_id == "noaa-gefs-forecast-35-day":
        paired = [
            unit
            for unit in units
            if unit.unit.tier.startswith("tier 1")
            and unit.complete is not None
            and {"0p25", "other"} <= unit.components.keys()
        ]
        flips = sum(
            unit.components["0p25"] > unit.components["other"] for unit in paired
        )
        lines.extend(
            [
                f"GEFS sibling-order flip rate (0p25 tier-1 input completes after 0p5): {flips}/{len(paired)} ({flips / len(paired):.2%})."
                if paired
                else "GEFS sibling-order flip rate: unmeasured (no complete paired inputs).",
                "",
            ]
        )
    return lines
