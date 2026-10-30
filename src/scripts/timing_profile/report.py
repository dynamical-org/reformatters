from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from zarr.storage import MemoryStore

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import ReformatCronJob
from reformatters.common.types import DatetimeLike
from reformatters.common.validation import CheckCurrentData
from reformatters.common.virtual_region_job import VirtualRegionJob

from .datasets import Unit, anchor_spacing, is_virtual
from .runtime import RuntimeStats
from .sources import Listing, listing_prefix
from .statistics import (
    MATERIALIZED_DEADLINE_CAP,
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
    virtual_poll_cutoff,
    virtual_poll_lead,
)
from .wxopticon import Evidence

# Dataset exceptions whose hand constants the rule must not replace, with the reason.
MANUAL_REVIEW_DATASETS = {
    "noaa-gefs-forecast-35-day-0-5-degree-virtual": "one daily polling window fires before the previous initialization's overnight extension finishes; a hand-set deadline keeps it polling through the newest morning wave",
    "noaa-mrms-conus-analysis-hourly": "fires near the radar-only p95, not at its T, because radar-only latency matters most to users; pod startup carries the first read past the radar-only p99",
}


@dataclass(frozen=True)
class MeasuredUnit:
    unit: Unit
    first: float | None
    complete: float | None
    missing_files: int
    file_arrivals: dict[frozenset[str], float]
    source: str = "listings"
    provenance: dict[str, int] = field(default_factory=dict)
    first_unmapped: str | None = None
    file_count: int | None = None

    @property
    def expected_files(self) -> int:
        return len(self.unit.files) if self.file_count is None else self.file_count


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
            evidence[url].provenance
            if evidence is not None
            else listings[listing_prefix(url)].provenance
            for url in file
        )
    visible = (
        [
            max(available[file] for file in group)
            for group in unit.visibility_groups
            if all(file in available for file in group)
        ]
        if unit.visibility_groups
        else available.values()
    )
    return MeasuredUnit(
        unit,
        min(visible, default=None),
        max(available.values()) if available and missing == 0 else None,
        missing,
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
) -> tuple[list[MeasuredUnit], pd.Timestamp | None, bool]:
    samples = [
        (unit.unit.anchor.timestamp(), value)
        for unit in units
        if (value := unit.first if virtual else unit.complete) is not None
    ]
    start, suspected = regime_start(samples, end.timestamp())
    boundary = pd.Timestamp(start, unit="s") if start is not None else None
    return (
        [unit for unit in units if boundary is None or unit.unit.anchor >= boundary],
        boundary,
        suspected,
    )


def small_sample(anchors: Sequence[pd.Timestamp]) -> bool:
    ordered = sorted(anchors)
    daily = (
        len(ordered) > 1
        and min((right - left).total_seconds() for left, right in pairwise(ordered))
        >= 86400
    )
    return len(ordered) < (300 if daily else 100)


def rate_arrivals(
    units: Sequence[MeasuredUnit], end: pd.Timestamp, horizon: float
) -> list[float | None]:
    return [
        unit.complete
        for unit in units
        if unit.missing_files < unit.expected_files
        and (
            unit.complete is not None
            or (end - unit.unit.anchor).total_seconds() > horizon
        )
    ]


def render_class_evidence(
    name: str,
    units: Sequence[MeasuredUnit],
    end: pd.Timestamp,
    *,
    virtual: bool,
) -> list[str]:
    active, boundary, suspected = class_history(units, end, virtual=virtual)
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
                f"**Regime restart at {boundary.isoformat()}Z:** retained {len(active)}/{len(units)} requested units. Whole-history segment medians differ by ≥15 min, or a trailing cluster exceeds the earlier max normal; the retained segment spans more than 24 hours.",
                "",
            ]
        )
    if suspected:
        lines.extend(
            [
                "**suspected change, review:** the terminal episode spans at most 24 hours; history is retained.",
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
    if small_sample(anchors):
        lines.extend(
            [
                f"**Small sample: n={len(samples)}; hand value needed.** Require ~100 units, or ~300 for daily classes, before using p99-derived constants.",
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
                "**tail-dominated, hand value needed:** p99 + 3 exceeds max normal + 3 by >15 min, or >1% of units exceed max normal. Materialized S/T and virtual max_delay are withheld for the affected readiness tier; first-visibility polling fires remain available.",
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


def fires_at(cron: ReformatCronJob, time: pd.Timestamp) -> bool:
    return cron.previous_fire_time(time) == time


def recommended_cron(offset: float, interval: float, cron: ReformatCronJob) -> str:
    rounded_minutes = round(offset / 60)
    minute = rounded_minutes % 60
    hour_shift = rounded_minutes // 60
    if interval < 3600:
        step = round(interval / 60)
        minutes = range(minute % step, 60, step)
        return f"{','.join(map(str, minutes))} * * * *"
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
    poll_lead: float | None = None
    completion_p99: float | None = None


def polling_startup(runtime: RuntimeStats) -> float:
    return runtime.startup_estimate if runtime.startup_estimate is not None else 0


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
    excluded: set[pd.Timestamp] | None = None,
) -> Recommendation:
    first = [unit.first for unit in units if unit.first is not None]
    if manual or not units:
        return Recommendation(None, None, None, None, ())
    end = (
        end
        if end is not None
        else max(unit.unit.anchor for unit in units) + pd.Timedelta(seconds=1)
    )
    tiers = list(dict.fromkeys(unit.unit.tier for unit in units))
    units = [unit for unit in units if unit.unit.anchor not in (excluded or set())]
    classes = []
    active_units = []
    completion_p99s = []
    for tier in tiers:
        for name, original in cycle_classes(
            [unit for unit in units if unit.unit.tier == tier], virtual=virtual
        ):
            grouped, _, _ = class_history(original, end, virtual=virtual)
            active_units.extend(grouped)
            if virtual:
                complete = [
                    unit.complete for unit in grouped if unit.complete is not None
                ]
                if complete:
                    completion_p99s.append(arrival_stats(complete).p99)
            arrivals = [
                value
                for unit in grouped
                if (value := unit.first if virtual else unit.complete) is not None
            ]
            if not arrivals:
                continue
            stats = arrival_stats(arrivals)
            threshold = schedule_offset(stats)
            offset = None if virtual else schedule_offset(stats)
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
                    tier,
                    name,
                    offset,
                    delay,
                    threshold,
                    tail_dominated(arrivals)
                    or small_sample(
                        [
                            unit.unit.anchor
                            for unit in grouped
                            if (unit.first if virtual else unit.complete) is not None
                        ]
                    ),
                )
            )
    completion_p99 = max(completion_p99s, default=None)
    poll_lead = None
    virtual_offset = None
    if virtual and first:
        startup = polling_startup(runtime)
        poll_lead = virtual_poll_lead(min(first), startup, interval, completion_p99)
        virtual_offset = virtual_fire(min(first), startup, poll_lead)
        classes = [replace(value, offset=virtual_offset) for value in classes]
    first_tier = [value for value in classes if value.tier == tiers[0]]
    offsets = [value.offset for value in first_tier if value.offset is not None]
    chosen = (
        (virtual_offset // 60) * 60
        if virtual_offset is not None
        else (None if virtual or not offsets else ceil_minute(max(offsets)))
    )
    if not virtual and any(value.hand_value for value in first_tier):
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
        else (None if virtual else min(interval, MATERIALIZED_DEADLINE_CAP))
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
        poll_lead,
        completion_p99,
    )


def render_dataset(
    dataset: DynamicalDataset[Any, Any],
    cron: ReformatCronJob,
    units: Sequence[MeasuredUnit],
    runtime: RuntimeStats,
    end: pd.Timestamp,
    gap: str | None,
    source_notes: set[str],
    *,
    excluded: set[pd.Timestamp] | None = None,
    triggered_by: Sequence[str] = (),
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
        *(
            [
                f"Triggered by {', '.join(triggered_by)}; the cron is a backstop at the first readiness tier's T, and max_delay {'matches' if current_delay is not None and fires_at(cron, pd.Timestamp(dataset.template_config.append_dim_start) + pd.Timedelta(seconds=current_delay)) else 'differs from'} its offset.",
                "",
            ]
            if triggered_by
            else []
        ),
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
        | {
            listing_prefix(unit.first_unmapped)
            for unit in units
            if not unit.unit.files and unit.first_unmapped is not None
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
    polling_units = units
    tiers = {
        tier: [
            unit
            for unit in units
            if unit.unit.tier == tier and unit.unit.anchor not in (excluded or set())
        ]
        for tier in dict.fromkeys(unit.unit.tier for unit in units)
    }
    if not tiers:
        lines.extend(
            [
                "Source: **not measured; keep constants** — no readiness units generated.",
                "",
            ]
        )
        return "\n".join(lines)
    manual = dataset.dataset_id in MANUAL_REVIEW_DATASETS
    recommendation = recommend(
        polling_units,
        runtime,
        virtual=virtual,
        interval=interval,
        currency_interval=currency_interval,
        spacing=anchor_spacing(dataset).total_seconds(),
        manual=manual,
        end=end,
        excluded=excluded,
    )
    if virtual:
        lines.extend(
            render_polling_evidence(
                polling_units, runtime, recommendation, interval, cron
            )
        )
    for tier, tier_units in tiers.items():
        lines.extend(
            [
                f"### {tier}",
                "",
                "| Cycle class | Active complete / expected | Missing old / recent censored | First usable file (virtual: first visibility) n / p50 / p95 / p99 | Unit p50 / p95 / p99 / T | Observed anchors (UTC) |",
                "|---|---:|---:|---|---|---|",
            ]
        )
        rates = []
        evidence_lines = []
        for cycle_class, original in cycle_classes(tier_units, virtual=virtual):
            evidence_lines.extend(
                render_class_evidence(cycle_class, original, end, virtual=virtual)
            )
            grouped, _, _ = class_history(original, end, virtual=virtual)
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
                eligible = rate_arrivals(grouped, end, horizon)
                empty = sum(
                    unit.complete is None
                    and unit.missing_files == unit.expected_files
                    and (end - unit.unit.anchor).total_seconds() > horizon
                    for unit in grouped
                )
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
                    f"| {cycle_class} | {duration(schedule_offset(stats) if stats else None)} | {duration(fire)} | {f'{repass:.2%}' if repass is not None else 'unmeasured'} | {f'{page:.2%}' if page is not None else 'unmeasured'} | {empty} |"
                )
        lines.extend(["", *evidence_lines])
        if rates:
            lines.extend(
                [
                    "| Cycle class | Class S/T = p99 + 3 min | Single-cron S used | Estimated re-pass rate | Estimated page rate | Older units with no observed files (excluded) |",
                    "|---|---:|---:|---:|---:|---:|",
                    *rates,
                    "",
                    "Estimates use p50 L/R/V: re-pass means incomplete at S + L; page means still incomplete at S + L + R + V + 5 min back-off + L. The first validation failure is quiet when a re-pass is queued. Recent incomplete units are censored. Unresolved units with an observed file count as incomplete; units with no observed files are excluded from the rate denominator and counted separately because retention is indistinguishable from nonarrival. Complete-unit lateness can overstate validation failures for multi-position units; trigger rescue is unmeasured. These are read-model estimates, not observed pages.",
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
        f"| {value.tier} / {value.cycle_class}{' (hand value needed)' if value.hand_value else ''} | {duration(value.offset) if virtual or not value.hand_value else 'hand value needed'} | {duration(ceil_minute(value.delay)) if not value.hand_value else 'hand value needed'} | {duration(value.threshold) if not value.hand_value else 'hand value needed'} |"
        for value in recommendation.classes
    )
    lines.extend(
        render_single_cron(
            dataset, cron, recommendation, units, interval, triggered=bool(triggered_by)
        )
    )
    return "\n".join(lines)


def render_polling_evidence(
    units: Sequence[MeasuredUnit],
    runtime: RuntimeStats,
    recommendation: Recommendation,
    interval: float,
    cron: ReformatCronJob,
) -> list[str]:
    first = [unit.first for unit in units if unit.first is not None]
    if not first:
        return []
    lines = [
        f"Polling E over the full requested window, all tiers/classes, without exclusions: {duration(min(first))}.",
        "",
    ]
    if recommendation.poll_lead is None or recommendation.offset is None:
        return lines
    startup = polling_startup(runtime)
    fifteen_minute_fire = (virtual_fire(min(first), startup) // 60) * 60
    cutoff = virtual_poll_cutoff(fifteen_minute_fire, interval, startup)
    completion_p99 = recommendation.completion_p99
    if completion_p99 is None:
        comparison = "has no completion p99 to compare"
    else:
        relation = "precedes" if cutoff < completion_p99 else "is at or after"
        comparison = f"{relation} maximum retained-class completion p99 {duration(completion_p99)}"
    startup_note = (
        ""
        if runtime.startup_estimate is not None
        else " L is unmeasured and assumed 0, so this is the latest acceptable fire."
    )
    lines.extend(
        [
            f"Polling lead: the 15-minute-lead fire {duration(fifteen_minute_fire)} polls until cutoff {duration(cutoff)}, which {comparison}, so the lead is {duration(recommendation.poll_lead)}.",
            "",
            f"Polling fire: E {duration(min(first))} - L p95 {duration(startup)} - lead {duration(recommendation.poll_lead)}, rounded down to the minute and clamped after the anchor: {duration(recommendation.offset)}; cron `{recommended_cron(recommendation.offset, interval, cron)}`; chosen cutoff {duration(virtual_poll_cutoff(recommendation.offset, interval, startup))}.{startup_note}",
            "",
        ]
    )
    if min(first) - startup - recommendation.poll_lead < 60:
        lines.extend(
            [
                "**Anchor guard:** polling fire clamped to the first minute after the source anchor: the template end is exclusive.",
                "",
            ]
        )
    return lines


def virtual_window_contains_anchor(
    dataset: DynamicalDataset[Any, Any], anchor: pd.Timestamp, offset: float
) -> bool:
    dim = dataset.template_config.append_dim

    def get_template(end: DatetimeLike) -> xr.DataTree:
        return xr.DataTree(
            xr.Dataset(
                coords={dim: dataset.template_config.append_dim_coordinates(end)}
            )
        )

    assert issubclass(dataset.region_job_class, VirtualRegionJob)
    jobs, _ = dataset.region_job_class.operational_update_jobs(
        MemoryStore(),
        Path(),
        get_template,
        dim,
        dataset.template_config.data_vars,
        "timing-profile",
        anchor + pd.Timedelta(seconds=offset),
    )
    return any(
        anchor in job.template_ds.to_dataset().get_index(dim)[job.region]
        for job in jobs
    )


def render_single_cron(
    dataset: DynamicalDataset[Any, Any],
    cron: ReformatCronJob,
    recommendation: Recommendation,
    units: Sequence[MeasuredUnit],
    interval: float,
    *,
    triggered: bool = False,
) -> list[str]:
    lines: list[str] = []
    virtual = is_virtual(dataset)
    chosen = recommendation.offset
    deadline = recommendation.deadline
    scalar_delay = recommendation.delay
    if chosen is not None:
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
                f"| max_delay | {duration(current_delay)} | {duration(scalar_delay) if scalar_delay is not None else 'hand value needed'} | {duration(scalar_delay - current_delay) if current_delay is not None and scalar_delay is not None else 'unmeasured'} |",
                f"| Pod deadline | {duration(cron.pod_active_deadline.total_seconds())} | {duration(ceil(deadline)) if deadline is not None else 'unmeasured; keep constants'} | {duration(ceil(deadline) - cron.pod_active_deadline.total_seconds()) if deadline is not None else 'unmeasured'} |",
                f"| Job deadline | {duration(cron.active_deadline.total_seconds()) if cron.active_deadline else 'unset'} | {duration(ceil(deadline)) if deadline is not None else 'unmeasured; keep constants'} | {duration(ceil(deadline) - cron.active_deadline.total_seconds()) if deadline is not None and cron.active_deadline else 'unset or unmeasured'} |",
            ]
        )
        lines.extend(
            [
                "",
                f"One-cron offset: {duration(chosen)}. {f'Uses first-visibility E across the full requested window, minus L and the {duration(recommendation.poll_lead)} polling lead, clamped after the anchor.' if virtual else f'Uses latest class and chosen {"backstop T" if triggered else "S"}; adoption remains manual.'}",
                "",
            ]
        )
        if virtual:
            if units and not virtual_window_contains_anchor(
                dataset, units[-1].unit.anchor, chosen
            ):
                lines.extend(
                    [
                        "**Window guard:** the proposed fire selects an earlier cycle through this dataset's window override. Keep constants or move the fire until the intended anchor is included.",
                        "",
                    ]
                )
            lines.extend(
                [
                    f"Validation grace: {duration(recommendation.validation_grace) if recommendation.validation_grace is not None else 'unmeasured; keep constants'} (maximum observed V plus allowance).",
                    "",
                ]
            )
    if dataset.dataset_id in MANUAL_REVIEW_DATASETS:
        lines.extend(
            [
                "",
                f"One-cron recommendation: **keep constants; review manually** (dataset exception: {MANUAL_REVIEW_DATASETS[dataset.dataset_id]}).",
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
