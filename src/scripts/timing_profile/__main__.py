"""Measure exact source-arrival units and recommend hand-set timing constants.

uv run python -m scripts.timing_profile --days 365 --output values-table.md

wxopticon's anonymous per-artifact history is preferred per complete mapped unit.
A unit with any unmapped artifact uses anonymous listings in full. Missing recent
units stay censored; an unavailable shard leaves other history usable. Use
--source listings to cross-check. A fresh --cache-dir captures a new frozen window.
Source-reported timestamps retain observed/reconstructed provenance; listings can
reflect overwrites. Published pooled summary percentiles are not imported.

S and backstop T use ceil_minute(p99 + 3 min). Use the longest single-regime
history available, at least 180 days; --days defaults to 365.
The max-normal diagnostic walks upward from the median to the first gap larger
than max(3 min, p90 - p50). A detected regime break restarts that class's history.
Tail-dominated and small-sample classes need hand values for p99-derived constants;
virtual polling fires from first-visibility E remain available. Short episodes flag review.
Re-pass/page rates are estimates from p50 L/R/V, not observed validation outcomes.

Runtimes (startup L, processing R, validation V) come only from Sentry check-ins:
set SENTRY_AUTH_TOKEN to a read-only Sentry token, or pass --checkins-dir with
offline monitor exports. An operator exports GET /api/0/organizations/<org>/
monitors/<slug>/checkins/, following Link's next cursor through the window, and
saves the concatenated JSON arrays as <slug>.json. Slugs are ReformatCronJob.name
and its -validate counterpart, not necessarily the dataset ID. Tokens are not saved.
An empty --checkins-dir leaves L/R/V unmeasured: deadlines, grace and re-pass/page
rates are not recommended, and virtual polling fires assume L = 0.

Default selection is every registered dataset whose update CronJob is not suspended;
suspended ones are listed and skipped. One run pins one --as-of (default: run start);
pass the same --as-of to every process of a multi-process capture. Concurrent runs
may share one --cache-dir. This advisory tool changes no dataset code and deploys
nothing.
"""

import argparse
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from reformatters.__main__ import DYNAMICAL_DATASETS, OPERATIONAL_ARCHIVERS
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import ReformatCronJob
from reformatters.common.logging import get_logger

from .datasets import expected_units, is_virtual, source_gap
from .report import aggregate_unit, render_dataset
from .runtime import RuntimeStats, fetch_checkins, measure_runtime
from .sources import TimestampSource, listing_prefix
from .wxopticon import WxopticonSource

log = get_logger(__name__)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Use the longest single-regime history available, at least 180 days; default 365.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "listings"],
        default="auto",
        help="Prefer exact wxopticon per-file history, falling back per unit; or force anonymous listings.",
    )
    parser.add_argument(
        "--as-of",
        type=pd.Timestamp,
        default=pd.Timestamp.now(tz="UTC").tz_convert(None),
        help="End of the window for every dataset in this run; default: run start.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset ID; repeat to select several. Default: every registered dataset whose update CronJob is not suspended.",
    )
    parser.add_argument(
        "--checkins-dir",
        type=Path,
        help="Offline Sentry check-in exports, <monitor slug>.json each; needed for L/R/V. Omit to fetch live with SENTRY_AUTH_TOKEN.",
    )
    parser.add_argument("--sentry-org", default="dynamical")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Frozen capture of wxopticon events and anonymous listings; reuse for the same window. Concurrent runs may share it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent anonymous listings; default 8.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Markdown here; default stdout. Results are flushed per dataset.",
    )
    parser.add_argument(
        "--exclude-anchor",
        action="append",
        default=[],
        metavar="DATASET=UTC_TIMESTAMP",
        help="Exclude a verified outlier from quantiles and rates, never polling E; repeat as needed.",
    )
    return parser.parse_args(argv)


def measure_source(
    dataset: DynamicalDataset[Any, Any],
    cron: ReformatCronJob,
    runtime: RuntimeStats,
    source: TimestampSource,
    start: pd.Timestamp,
    end: pd.Timestamp,
    excluded: set[pd.Timestamp],
    *,
    source_mode: Literal["auto", "listings"] = "auto",
    triggered_by: Sequence[str] = (),
) -> str:
    gap = source_gap(dataset.dataset_id)
    weather_next = "weathernext" in dataset.dataset_id
    if weather_next and not gap:
        config = dataset.template_config
        frequency = config.append_dim_frequency
        anchors = pd.date_range(
            start.ceil(frequency), end, freq=frequency, inclusive="left"
        )
        pages = (
            len(anchors)
            * sum(
                50 if var.group == "pressure_level" else 4 for var in config.data_vars
            )
            if "weathernext2" in dataset.dataset_id
            else len(config.data_vars)
            * sum(7 if anchor.hour % 6 == 0 else 1 for anchor in anchors)
        )
        log.info(
            "%s: uncached proxy capture estimate ~%d requests, at least %.1f minutes "
            "at 2 requests/s using observed page costs; actual paging can vary. "
            "Size --days to the pacing budget.",
            dataset.dataset_id,
            pages,
            pages / 120,
        )
    pending = iter(()) if gap else expected_units(dataset, start, end, source)
    batches = (
        groupby(pending, key=lambda unit: unit.anchor)
        if weather_next
        else ((None, pending),)
    )
    measured = []
    source_notes = set()
    started = time.monotonic()
    proxy_requests = source.proxy_requests
    for anchor, batch in batches:
        units = list(batch)
        evidence, source_note = (
            WxopticonSource(source).measure(dataset.dataset_id, units)
            if source_mode == "auto" and not gap
            else ({}, "anonymous listings explicitly selected")
        )
        source_notes.add(source_note)
        mapped = [
            all(url in evidence for file in unit.files for url in file)
            for unit in units
        ]
        prefixes = {
            listing_prefix(url)
            for unit, covered in zip(units, mapped, strict=True)
            if not covered
            for file in unit.files
            for url in file
        }
        log.info(
            "%s %s: %d readiness units, %d listing prefixes",
            dataset.dataset_id,
            anchor or "",
            len(units),
            len(prefixes),
        )
        listings = source.list_many(prefixes)
        source_notes.update(listing.note for listing in listings.values())
        for unit, covered in zip(units, mapped, strict=True):
            result = aggregate_unit(
                unit,
                listings,
                as_of=end,
                evidence=evidence if covered else None,
                first_unmapped=next(
                    (
                        url
                        for file in unit.files
                        for url in sorted(file)
                        if url not in evidence
                    ),
                    None,
                )
                if not covered
                else None,
            )
            if weather_next:
                result = replace(
                    result,
                    unit=replace(unit, files=()),
                    file_count=len(unit.files),
                    file_arrivals={},
                )
            measured.append(result)
    if weather_next and not gap:
        source_notes.add(
            f"proxy capture: {source.proxy_requests - proxy_requests} requests in "
            f"{time.monotonic() - started:.1f}s elapsed, paced at at most 2 requests/s; "
            "capture duration is not publication latency"
        )
    return render_dataset(
        dataset,
        cron,
        measured,
        runtime,
        end,
        gap,
        source_notes,
        excluded=excluded,
        triggered_by=triggered_by,
    )


def parse_exclusions(entries: Sequence[str]) -> dict[str, set[pd.Timestamp]]:
    excluded: dict[str, set[pd.Timestamp]] = {}
    for entry in entries:
        dataset_id, timestamp = entry.split("=", 1)
        anchor = pd.Timestamp(timestamp)
        excluded.setdefault(dataset_id, set()).add(
            anchor.tz_convert(None) if anchor.tzinfo else anchor
        )
    return excluded


def select_datasets(
    requested: Sequence[str] | None,
) -> tuple[list[tuple[DynamicalDataset[Any, Any], ReformatCronJob]], list[str]]:
    """Datasets to profile with their update CronJobs, and the suspended ones skipped.

    An explicit request profiles the named datasets even when suspended."""
    selected = []
    suspended = []
    for dataset in DYNAMICAL_DATASETS:
        if requested and dataset.dataset_id not in requested:
            continue
        cron = next(
            (
                resource
                for resource in dataset.operational_kubernetes_resources(
                    "timing-profile"
                )
                if isinstance(resource, ReformatCronJob)
            ),
            None,
        )
        if cron is None:
            continue
        if cron.suspend and not requested:
            suspended.append(dataset.dataset_id)
        else:
            selected.append((dataset, cron))
    assert not requested or set(requested) <= {
        dataset.dataset_id for dataset in DYNAMICAL_DATASETS
    }, "Unknown dataset ID"
    return selected, suspended


def trigger_producers() -> dict[str, list[str]]:
    producers: dict[str, list[str]] = {}
    for resource in (*DYNAMICAL_DATASETS, *OPERATIONAL_ARCHIVERS):
        for trigger in resource.materialized_update_triggers:
            producers.setdefault(trigger.target_cronjob, []).append(resource.dataset_id)
    return producers


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    assert args.days > 0
    assert args.workers > 0
    end = args.as_of
    if end.tzinfo is not None:
        end = end.tz_convert(None)
    start = end - pd.Timedelta(days=args.days)
    capture = args.cache_dir or Path(tempfile.mkdtemp(prefix="timing-profile-"))
    checkins_dir = args.checkins_dir
    if checkins_dir is None:
        assert os.environ.get("SENTRY_AUTH_TOKEN"), (
            "Set SENTRY_AUTH_TOKEN or pass --checkins-dir"
        )
        checkins_dir = capture / "checkins"
    excluded = parse_exclusions(args.exclude_anchor)
    selected, suspended = select_datasets(args.dataset)
    output = args.output.open("w") if args.output else sys.stdout
    output.write(
        f"# Operational timing measurements\n\nGenerated {datetime.now(UTC).isoformat()}. Requested window: {start} .. {end} UTC ({args.days} days). Anonymous capture: `{capture}`. Source selection: {args.source}. Check-ins: {'offline exports at ' + str(checkins_dir) if args.checkins_dir else 'live read-only Sentry GET, captured at ' + str(checkins_dir)}. Verified outlier anchors excluded from quantiles/rates (never E): {args.exclude_anchor or 'none'}.\n\n"
        "Durations are hours:minutes:seconds after the anchor unless stated otherwise. Quantiles use linear interpolation. Writes after as-of are excluded. Each unit uses wxopticon only if every required artifact maps; otherwise the entire unit uses listings. Coverage and observed/reconstructed provenance appear per class. Missing recent units remain censored; older missing objects can reflect retention. A usable file includes its index; a complete unit includes every required file/member. First visibility for gated forecast-derived virtual analyses uses own-cycle inputs; polling E is the earliest first visibility. Forecast-derived analysis units include their accumulated boundary input. Daily observations anchor at valid-period end.\n\n"
        "S and backstop T = ceil_minute(p99 + 3 min), with the single materialized cron at the latest first-tier class. Use the longest single-regime history available, at least 180 days (default 365). Search the whole history for the two-segment split minimizing absolute deviations around the segment medians, with at least five samples on each side; restart when their medians differ by ≥15 min. A trailing cluster above earlier max normal (at least two units and ≥1% of observations) can also restart history. A retained segment must span more than 24 hours; otherwise flag a suspected change for review and keep the history. Repeat the fit on each retained segment until no further qualifying restart is found. The max-normal cut is the first gap above the median larger than max(3 min, p90 - p50). Compare p99 + 3 with max normal + 3; tail-dominated and small-sample classes (<100 units, or <300 for daily classes) need hand values for p99-derived constants. Virtual polling uses first-visibility E over the entire requested window without exclusions or regime trimming, minus L and 15 min (3 min when that fire's poll cutoff precedes a retained class's complete-arrival p99), clamped to the first minute after the anchor (exclusive template end); short histories are flagged.\n\n"
        "Re-pass/page estimates use measured p50 L/R/V: first read S + L; first re-pass read S + L + R + V + 5 min back-off + L. A first validation failure is quiet when a re-pass is queued. Recent incomplete units are censored. Unresolved units with an observed file are counted as incomplete; units with no observed files are excluded from rate denominators and counted separately. These complete-unit estimates can overstate validation failures for multi-position units; trigger rescue is unmeasured. L p95 remains the startup estimate for virtual polling, R p95 covers scheduled-ok materialized processing, and V max bounds validation. Historical check-in schedules classify scheduled runs approximately; starts ≥5 min after fire are excluded, and check-ins do not establish which source init was processed. Missing runtime components leave rates unmeasured. Datasets marked for manual review keep hand constants. No production constants are changed.\n\n"
    )
    output.flush()
    producers = trigger_producers()
    source = TimestampSource(capture / "sources", args.workers)
    try:
        for dataset, cron in selected:
            log.info("Measuring %s", dataset.dataset_id)
            if args.checkins_dir is None:
                for monitor in (
                    cron.name,
                    cron.name.removesuffix("-update") + "-validate",
                ):
                    fetch_checkins(checkins_dir, monitor, args.sentry_org, start)
            runtime = measure_runtime(
                checkins_dir, cron, start, end, virtual=is_virtual(dataset)
            )
            output.write(
                measure_source(
                    dataset,
                    cron,
                    runtime,
                    source,
                    start,
                    end,
                    excluded.get(dataset.dataset_id, set()),
                    source_mode=args.source,
                    triggered_by=producers.get(cron.name, []),
                )
                + "\n"
            )
            output.flush()
        if suspended:
            output.write(
                "## Suspended, not profiled\n\n"
                + "".join(f"- {dataset_id}\n" for dataset_id in suspended)
                + "\n"
            )
        if not args.dataset:
            output.write("## Operational archivers\n\n")
            for archiver in OPERATIONAL_ARCHIVERS:
                output.write(
                    f"- {archiver.dataset_id}: upstream first-publication timing not measured; keep constants. Archive copy timestamps measure archive readiness, not upstream availability.\n"
                )
    finally:
        source.close()
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
