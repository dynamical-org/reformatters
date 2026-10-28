from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
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
from .wxopticon import Evidence

NAMED_EXCEPTIONS = frozenset(
    {
        "noaa-gefs-forecast-35-day-0-5-degree-virtual",
        "noaa-mrms-conus-analysis-hourly",
    }
)


@dataclass(frozen=True)
class MeasuredUnit:
    unit: Unit
    first: float | None
    complete: float | None
    missing_files: int
    raw_first: float | None
    file_arrivals: dict[frozenset[str], float]
    source: str = "listings"
    provenance: dict[str, int] = field(default_factory=dict)
    first_unmapped: str | None = None


def aggregate_unit(
    unit: Unit,
    listings: dict[str, Listing],
    *,
    as_of: pd.Timestamp | None = None,
    evidence: dict[str, Evidence] | None = None,
    first_unmapped: str | None = None,
) -> MeasuredUnit:
    available: dict[frozenset[str], float] = {}
    provenance: Counter[str] = Counter()
    missing = 0
    for file in unit.files:
        timestamps = [
            evidence[url].arrival
            if evidence is not None
            else listings[listing_prefix(url)].objects.get(url)
            for url in file
        ]
        timestamps = [
            value if as_of is None or value is None or value <= as_of else None
            for value in timestamps
        ]
        if any(value is None for value in timestamps):
            missing += 1
            continue
        usable = max(value for value in timestamps if value is not None)
        offset = (usable - unit.anchor).total_seconds()
        available[file] = offset
        provenance.update(
            evidence[url].provenance if evidence is not None else "current_object_write"
            for url in file
        )
    raw_first = min(available.values()) if available else None
    visible = [
        max(available[file] for file in group)
        for group in unit.visibility_groups
        if all(file in available for file in group)
    ]
    return MeasuredUnit(
        unit,
        (min(visible) if visible else None) if unit.visibility_groups else raw_first,
        max(available.values()) if available and missing == 0 else None,
        missing,
        raw_first,
        available,
        "wxopticon" if evidence is not None else "listings",
        dict(provenance),
        first_unmapped,
    )


def duration(value: float | None) -> str:
    if value is None:
        return "unmeasured"
    sign = "-" if value < 0 else ""
    seconds = round(abs(value))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{sign}{hours:02}:{minutes:02}:{seconds:02}"


def class_history(
    units: Sequence[MeasuredUnit],
    end: pd.Timestamp,
    *,
    virtual: bool,
) -> tuple[list[MeasuredUnit], pd.Timestamp | None]:
    samples = [
        (unit.unit.anchor.timestamp(), value)
        for unit in units
        if (value := unit.first if virtual else unit.complete) is not None
    ]
    start = regime_start(samples, end.timestamp())
    boundary = pd.Timestamp(start, unit="s") if start is not None else None
    return [
        unit for unit in units if boundary is None or unit.unit.anchor >= boundary
    ], boundary


def render_class_evidence(
    name: str,
    units: Sequence[MeasuredUnit],
    end: pd.Timestamp,
    *,
    virtual: bool,
) -> list[str]:
    active, boundary = class_history(units, end, virtual=virtual)
    samples = [
        (unit, value)
        for unit in active
        if (value := unit.first if virtual else unit.complete) is not None
    ]
    sources = ", ".join(sorted({unit.source for unit in units}))
    mapped = sum(unit.source == "wxopticon" for unit in units)
    first_unmapped = next(
        (unit.first_unmapped for unit in units if unit.first_unmapped), "not captured"
    )
    provenance: Counter[str] = Counter()
    for unit in active:
        provenance.update(unit.provenance)
    mix = (
        ", ".join(f"{key}={count}" for key, count in sorted(provenance.items()))
        or "unmeasured"
    )
    anchors = sorted(unit.unit.anchor for unit, _ in samples)
    days = len({anchor.normalize() for anchor in anchors})
    span = (anchors[-1] - anchors[0]).days + 1 if anchors else 0
    lines = [
        f"Evidence {name}: source **{sources}**; wxopticon {mapped}/{len(units)} units; listings for the rest (first unmapped: {first_unmapped if mapped < len(units) else 'none'}). Active history {days} sampled days / {span} calendar days; n={len(samples)}. Provenance (required artifacts in active history): {mix}.",
        "",
    ]
    if boundary is not None:
        lines.extend(
            [
                f"**Regime restart at {boundary.isoformat()}Z:** retained {len(active)}/{len(units)} requested units. Last-14-day median shift ≥15 min or a trailing cluster above the earlier max normal; statistics and recommendations use this history.",
                "",
            ]
        )
    if days < 180:
        lines.extend(
            [
                "**Short history (<180 sampled days); p99 needs at least ~180 days per regime.**",
                "",
            ]
        )
    daily = (
        len(anchors) > 1
        and min((right - left).total_seconds() for left, right in pairwise(anchors))
        >= 86400
    )
    if len(samples) < (300 if daily else 100):
        lines.extend(
            [
                f"**Small sample: n={len(samples)}; need ~{'300 for daily classes' if daily else '100 units'} before relying on p99.**",
                "",
            ]
        )
    if not samples:
        return lines
    values = [value for _, value in samples]
    cut = max_normal_cut(values)
    outliers = (
        "; ".join(
            f"{unit.unit.anchor.isoformat()}Z ({duration(value)}; {unit.source}; {', '.join(sorted(unit.provenance))})"
            for unit, value in sorted(samples, key=lambda pair: pair[0].unit.anchor)
            if value > cut
        )
        or "none"
    )
    metric = "first visibility" if virtual else "complete unit"
    lines.extend(
        [
            f"Provisional max-normal cut ({metric}): {duration(cut)}. First gap above the median larger than g = max(3 min, p90 - p50) = {duration(max_normal_gap(values))}. Above-cut anchors (UTC): {outliers}.",
            "",
            f"p99 + 3 min: {duration(schedule_offset(arrival_stats(values)))}; max normal + 3 min: {duration(ceil_minute(cut + 3 * 60))}. Both rounded up to the fire minute. Beyond max normal: {sum(value > cut for value in values) / len(values):.2%}.",
            "",
        ]
    )
    if tail_dominated(values):
        lines.extend(
            [
                "**tail-dominated, hand value needed:** p99 + 3 exceeds max normal + 3 by >15 min, or >1% of units exceed max normal. Automatic constants are withheld for the affected readiness tier.",
                "",
            ]
        )
    return lines


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
    *,
    virtual: bool = False,
) -> list[tuple[str, list[MeasuredUnit]]]:
    by_hour: dict[int, list[MeasuredUnit]] = defaultdict(list)
    for unit in units:
        by_hour[unit.unit.anchor.hour].append(unit)
    arrivals = {
        hour: [
            value
            for unit in values
            if (value := unit.first if virtual else unit.complete) is not None
        ]
        for hour, values in by_hour.items()
    }
    measured = {
        hour: arrival_stats(values).p50 for hour, values in arrivals.items() if values
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


@dataclass(frozen=True)
class ClassRecommendation:
    tier: str
    cycle_class: str
    offset: float | None
    delay: float
    threshold: float
    hand_value: bool = False


@dataclass(frozen=True)
class Recommendation:
    offset: float | None
    delay: float | None
    deadline: float | None
    validation_grace: float | None
    classes: tuple[ClassRecommendation, ...]


def recommend(
    units: Sequence[MeasuredUnit],
    runtime: RuntimeStats,
    *,
    virtual: bool,
    interval: float,
    currency_interval: float,
    spacing: float,
    manual: bool = False,
    end: pd.Timestamp | None = None,
) -> Recommendation:
    if manual or not units:
        return Recommendation(None, None, None, None, ())
    end = (
        end
        if end is not None
        else max(unit.unit.anchor for unit in units) + pd.Timedelta(seconds=1)
    )
    tiers = list(dict.fromkeys(unit.unit.tier for unit in units))
    classes = []
    active_units = []
    for tier in tiers:
        for name, original in cycle_classes(
            [unit for unit in units if unit.unit.tier == tier], virtual=virtual
        ):
            grouped, _ = class_history(original, end, virtual=virtual)
            active_units.extend(grouped)
            arrivals = [
                value
                for unit in grouped
                if (value := unit.first if virtual else unit.complete) is not None
            ]
            if not arrivals:
                continue
            stats = arrival_stats(arrivals)
            threshold = schedule_offset(stats)
            raw = [unit.raw_first for unit in grouped if unit.raw_first is not None]
            offset = (
                virtual_fire(min(raw), runtime.startup_estimate)
                if virtual and raw and runtime.startup_estimate is not None
                else (None if virtual else schedule_offset(stats))
            )
            if not virtual:
                assert offset is not None
            delay = currency_delay(
                threshold if offset is None or virtual else ceil_minute(offset),
                grouped[0].unit.anchor_minus_label,
                currency_interval,
                spacing,
                virtual=virtual,
            )
            classes.append(
                ClassRecommendation(
                    tier, name, offset, delay, threshold, tail_dominated(arrivals)
                )
            )
    first_tier = [value for value in classes if value.tier == tiers[0]]
    offsets = [value.offset for value in first_tier if value.offset is not None]
    raw = [unit.raw_first for unit in active_units if unit.raw_first is not None]
    chosen = (
        (virtual_fire(min(raw), runtime.startup_estimate) // 60) * 60
        if virtual and raw and runtime.startup_estimate is not None
        else (None if virtual or not offsets else ceil_minute(max(offsets)))
    )
    if any(value.hand_value for value in first_tier):
        chosen = None
    delays = (
        [value.delay for value in first_tier]
        if virtual
        else [
            currency_delay(
                chosen,
                unit.unit.anchor_minus_label,
                currency_interval,
                spacing,
                virtual=False,
            )
            for unit in active_units
            if unit.unit.tier == tiers[0] and chosen is not None
        ]
    )
    deadline = (
        virtual_deadline(interval, runtime.startup_estimate, runtime.validation_max)
        if virtual
        and runtime.startup_estimate is not None
        and runtime.validation_max is not None
        else (None if virtual else interval)
    )
    return Recommendation(
        chosen,
        ceil_minute(max(delays))
        if delays and not any(value.hand_value for value in first_tier)
        else None,
        deadline,
        ceil(runtime.validation_max + 30)
        if virtual and runtime.validation_max is not None
        else None,
        tuple(classes),
    )


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
        f"Runtime p50 (re-pass/page model): L {duration(runtime.startup_p50)}; R {duration(runtime.processing_p50)}; V {duration(runtime.validation_p50)}. L estimate p95 {duration(runtime.startup_estimate)}; R p95 {duration(runtime.processing)}; V p95/max {duration(runtime.validation_p95)} / {duration(runtime.validation_max)}. Scheduled-ok n={runtime.scheduled_count}; matched inline validation n={runtime.validation_count}; non-ok update n={runtime.non_ok_count}; off-schedule ok n={runtime.off_schedule_count}. Check-ins: {runtime.first} .. {runtime.last} UTC.",
        "",
    ]
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
    manual = dataset.dataset_id in NAMED_EXCEPTIONS
    recommendation = recommend(
        units,
        runtime,
        virtual=virtual,
        interval=interval,
        currency_interval=currency_interval,
        spacing=anchor_spacing(dataset).total_seconds(),
        manual=manual,
        end=end,
    )
    for tier, tier_units in tiers.items():
        lines.extend(
            [
                f"### {tier}",
                "",
                "| Cycle class | Active complete / expected | Missing old / recent censored | First usable file (virtual: first visibility) n / p50 / p95 / p99; raw E | Unit p50 / p95 / p99 / T | Observed anchors (UTC) |",
                "|---|---:|---:|---|---|---|",
            ]
        )
        rates = []
        evidence_lines = []
        for cycle_class, original in cycle_classes(tier_units, virtual=virtual):
            evidence_lines.extend(
                render_class_evidence(cycle_class, original, end, virtual=virtual)
            )
            grouped, _ = class_history(original, end, virtual=virtual)
            complete = [unit.complete for unit in grouped if unit.complete is not None]
            first = [unit.first for unit in grouped if unit.first is not None]
            horizon = max(
                complete,
                default=max(
                    (unit.complete for unit in tier_units if unit.complete is not None),
                    default=0,
                ),
            )
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
                    )
                )
                + "; "
                + duration(
                    min(raw)
                    if (
                        raw := [u.raw_first for u in grouped if u.raw_first is not None]
                    )
                    else None
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
                        schedule_offset(
                            first_stats
                            if virtual and first_stats is not None
                            else stats
                        ),
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
            if not manual and not virtual:
                fire = recommendation.offset
                eligible = [
                    unit.complete
                    for unit in grouped
                    if unit.complete is not None
                    or (end - unit.unit.anchor).total_seconds() > horizon
                ]
                repass, page = (
                    read_rates(
                        eligible,
                        fire,
                        runtime.startup_p50,
                        runtime.processing_p50,
                        runtime.validation_p50,
                    )
                    if fire is not None
                    else (None, None)
                )
                rates.append(
                    f"| {cycle_class} | {duration(schedule_offset(stats) if stats else None)} | {duration(fire)} | {f'{repass:.2%}' if repass is not None else 'unmeasured'} | {f'{page:.2%}' if page is not None else 'unmeasured'} | {missing} |"
                )
        lines.extend(["", *evidence_lines])
        if rates:
            lines.extend(
                [
                    "| Cycle class | Class S/T = p99 + 3 min | Single-cron S used | Estimated re-pass rate | Estimated page rate | Older unresolved units included |",
                    "|---|---:|---:|---:|---:|---:|",
                    *rates,
                    "",
                    "Estimates use p50 L/R/V: re-pass means incomplete at S + L; page means still incomplete at S + L + R + V + 5 min back-off + L. The first validation failure is quiet when a re-pass is queued. Recent incomplete units are censored; older unresolved units are included as incomplete and can reflect source retention. Complete-unit lateness can overstate validation failures for multi-position units; trigger rescue is unmeasured. These are read-model estimates, not observed pages.",
                    "",
                ]
            )
    lines.extend(
        [
            "### Advisory constants",
            "",
            "| Tier / class | Per-class offset | max_delay | T (first visibility for virtuals; complete unit otherwise) |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {value.tier} / {value.cycle_class}{' (hand value needed)' if value.hand_value else ''} | {duration(value.offset) if not value.hand_value else 'hand value needed'} | {duration(ceil_minute(value.delay)) if not value.hand_value else 'hand value needed'} | {duration(value.threshold) if not value.hand_value else 'hand value needed'} |"
        for value in recommendation.classes
    )
    lines.extend(render_single_cron(dataset, cron, recommendation, units, interval))
    return "\n".join(lines)


def render_single_cron(
    dataset: DynamicalDataset[Any, Any],
    cron: ReformatCronJob,
    recommendation: Recommendation,
    units: Sequence[MeasuredUnit],
    interval: float,
) -> list[str]:
    lines: list[str] = []
    virtual = is_virtual(dataset)
    chosen = recommendation.offset
    deadline = recommendation.deadline
    scalar_delay = recommendation.delay
    if chosen is not None and scalar_delay is not None:
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
                f"One-cron offset: {duration(chosen)}. {'Uses earliest class and first-file E.' if virtual else 'Uses latest class and chosen S; adoption remains manual.'}",
                "",
            ]
        )
        if virtual:
            lines.extend(
                [
                    f"Validation grace: {duration(recommendation.validation_grace) if recommendation.validation_grace is not None else 'unmeasured; keep constants'} (maximum observed V plus allowance).",
                    "",
                ]
            )
    if dataset.dataset_id in NAMED_EXCEPTIONS:
        lines.extend(
            [
                "",
                "One-cron recommendation: **keep constants; review manually** (named exception).",
                "",
            ]
        )
    if dataset.dataset_id == "noaa-gefs-forecast-35-day":
        paired = []
        for unit in units:
            if not unit.unit.tier.startswith("tier 1") or unit.complete is None:
                continue
            siblings: dict[bool, list[float]] = defaultdict(list)
            for file in unit.unit.files:
                siblings[any(".0p25." in url for url in file)].append(
                    unit.file_arrivals[file]
                )
            if siblings[True] and siblings[False]:
                paired.append((max(siblings[True]), max(siblings[False])))
        flips = sum(
            high_resolution > low_resolution
            for high_resolution, low_resolution in paired
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
