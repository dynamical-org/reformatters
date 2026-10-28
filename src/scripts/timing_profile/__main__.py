"""Measure source arrivals and recommend hand-set operational timing constants.

Run with `uv run python -m scripts.timing_profile --days 35 --output values-table.md`.
This is a local, advisory tool: it does not change dataset code or deploy anything.

For live runtimes, set SENTRY_AUTH_TOKEN to a read-only Sentry token. The tool
makes only GET requests, and never persists the token. Alternatively, export each
update and validate monitor's check-ins from Sentry's
GET /api/0/organizations/<org>/monitors/<slug>/checkins/ endpoint. Follow Link's
next cursor until the requested window is covered, concatenate the JSON arrays,
and save one <slug>.json file per monitor. Pass that directory to --checkins-dir.
The monitor slugs come from the dataset's ReformatCronJob.name and its -validate
counterpart, not necessarily the dataset ID.

Anonymous S3 LastModified, GCS updated and HTTP directory Last modified timestamps
are evidence of the current object write, not necessarily first publication. Review overwrites, short retention, small samples and regime
changes before adopting constants. Use a fresh --cache-dir for each new window;
a cache is a frozen capture, including missing objects. E is the observed minimum;
exclude only verified outlier anchors explicitly, retaining recurring fast periods.

wxopticon's anonymous per-file archive is preferred when every required artifact
in the requested --days window maps exactly. Otherwise the whole dataset uses
anonymous listings, with the coverage gap reported. Mirrors and grouped events
are not substituted for exact files. Use --source listings for a cross-check.
Use --days 365 for long history where coverage permits. Reports show provenance,
short histories, a provisional largest-gap max-normal cut and recent median shifts.
Published summary percentiles pool cycles; this tool derives classes from events.
"""

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    parser.add_argument("--days", type=int, default=35)
    parser.add_argument(
        "--source",
        choices=["auto", "listings"],
        default="auto",
        help="Prefer exact wxopticon per-file history, falling back per dataset; or force anonymous listings.",
    )
    parser.add_argument(
        "--as-of",
        type=pd.Timestamp,
        default=pd.Timestamp.now(tz="UTC").tz_convert(None),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset ID; repeat to select several. Default: every registered dataset with an update CronJob, including suspended datasets.",
    )
    parser.add_argument(
        "--checkins-dir",
        type=Path,
        help="Offline check-in JSON exports. Omit to fetch live with SENTRY_AUTH_TOKEN.",
    )
    parser.add_argument("--sentry-org", default="dynamical")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Frozen capture of wxopticon events and anonymous listings; reuse for the same window.",
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
        help="Exclude a verified outlier anchor from arrival statistics; repeat as needed.",
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
    source_mode: str = "auto",
) -> str:
    gap = source_gap(dataset.dataset_id)
    if "weathernext" in dataset.dataset_id and gap is None:
        bucket = (
            "weathernext3_statistics_spatial"
            if "weathernext3" in dataset.dataset_id
            else "weathernext"
        )
        gap = source.probe_gcs(bucket)
    units = [] if gap else list(expected_units(dataset, start, end, source))
    units = [unit for unit in units if unit.anchor not in excluded]
    evidence, source_note = (
        WxopticonSource(source).measure(dataset.dataset_id, units)
        if source_mode == "auto" and not gap
        else (None, "anonymous listings explicitly selected")
    )
    prefixes = (
        {listing_prefix(url) for unit in units for file in unit.files for url in file}
        if evidence is None
        else set()
    )
    log.info(
        "%s: %d readiness units, %d listing prefixes",
        dataset.dataset_id,
        len(units),
        len(prefixes),
    )
    listings = source.list_many(prefixes)
    measured = [
        aggregate_unit(unit, listings, as_of=end, evidence=evidence) for unit in units
    ]
    return render_dataset(
        dataset,
        cron,
        measured,
        runtime,
        end,
        gap,
        {source_note, *(listing.note for listing in listings.values())},
    )


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
    excluded: dict[str, set[pd.Timestamp]] = {}
    for entry in args.exclude_anchor:
        dataset_id, timestamp = entry.split("=", 1)
        anchor = pd.Timestamp(timestamp)
        excluded.setdefault(dataset_id, set()).add(
            anchor.tz_convert(None) if anchor.tzinfo else anchor
        )
    selected = [
        dataset
        for dataset in DYNAMICAL_DATASETS
        if not args.dataset or dataset.dataset_id in args.dataset
    ]
    assert not args.dataset or set(args.dataset) <= {
        ds.dataset_id for ds in selected
    }, "Unknown dataset ID"
    output = args.output.open("w") if args.output else sys.stdout
    output.write(
        f"# Operational timing measurements\n\nGenerated {datetime.now(UTC).isoformat()}. Requested window: {start} .. {end} UTC ({args.days} days). Anonymous source capture: `{capture}`. Check-ins: {'offline exports at ' + str(checkins_dir) if args.checkins_dir else 'live read-only Sentry GET, captured at ' + str(checkins_dir)}. Excluded verified outlier anchors: {args.exclude_anchor or 'none'}.\n\nAll durations are hours:minutes:seconds after the anchor unless stated otherwise. Quantiles use linear interpolation. Object writes after the requested as-of time are excluded, even if collected later. Source identity and observed coverage appear per dataset. Source selection: {args.source}; wxopticon requires exact coverage of the entire requested file set, otherwise listings are used for the dataset. Provenance counts distinguish directly_observed from reconstructed_authoritative; neither implies observation time is the publication timestamp. A provisional max-normal cut uses the largest adjacent gap above the median and leaves every sample in the percentiles. Recent shifts compare last-7-day and preceding-28-day medians (at least 5 samples each, 15-minute difference). A usable file includes its required index; a complete unit includes every required file/member. First visibility is min(files) for forecasts; gated HRRR/GFS/GEFS analysis virtuals use min over positions of max(own-cycle files). Raw E remains min(files) for polling. Virtual cycle classes split on first visibility; materialized classes split on complete-unit arrival. Tier order follows expected-unit generation. MRMS pass 1 is printed separately; its hourly schedule remains a named exception. GFS, GEFS and materialized HRRR analyses include the accumulated boundary input required by their final trim. Daily observations use the end of the valid day as anchor. GEFS 35-day tiers are separate; expected counts include only units the dataset coordinate factory emits, so age-gated units not yet eligible can be absent. WeatherNext units contain every required member/level of a variable and lead.\n\nL is a schedule-to-check-in estimate including scheduling, imports and initialization. R is scheduled-ok materialized update duration through publish; virtual R is unmeasured because update duration includes source waiting. V includes completed inline validations (ok or error) matched within 120 seconds after update completion; p95 and maximum are both shown. Failed/killed updates and validations without a closing duration are censored from durations. Runs starting at least 5 minutes after the previous fire are excluded as likely manual/retry/trigger runs; classification from timestamps is approximate. Historical per-check-in schedules are used, never expectedTime. Runtime statistics pool the available runs; check-ins do not establish which source initialization an update processed. The validation maximum includes matched manual/retry/trigger runs.\n\nS candidates are advisory. The chosen S is min(p95 + max(p95 - p50, 5 min), p99 + 1 min); materialized cron and currency recommendations use it. Fraction of units arriving after S uses complete observed units only, excluding missing/censored units; short retention can bias it. After S or never includes older missing units in numerator and denominator, excluding recent censored units; missing objects can mean expired retention rather than never published. Recent censoring denotes an incomplete anchor no older than the slowest complete observed unit; older absent units can reflect missing data or source retention, which listing timestamps cannot distinguish. Median latency estimates cover units ready by S only; late-unit re-pass timing is unmeasured. Multi-position units can pass validation with a complete prefix. E includes every observed first-file arrival except explicit exclusions; inspect the lower tail before adoption. Named exception constants require manual review. Primary operational sources are measured; mirror fallback timing is unmeasured unless shown as its own dataset source. No production constants are changed.\n\n"
    )
    output.flush()
    source = TimestampSource(capture / "sources", args.workers)
    try:
        for dataset in selected:
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
                )
                + "\n"
            )
            output.flush()
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
