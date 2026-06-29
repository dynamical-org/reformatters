"""Whole-archive manifest completeness scan for a virtual dataset.

The offline analog of the operational `CheckVirtualManifestCompleteness`: instead of a
recent window it probes every expected source file across the archive for ref existence
(no decode), reporting per-append-dim-position availability and emitting backfill retry
filters for any gaps. This is the thorough post-backfill availability gate; the value-based
null scan in report_nulls is skipped on virtual stores because presence is a manifest
question, not a value question. See docs/validation.md.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import zarr
from icechunk.store import IcechunkStore

from reformatters.common.logging import get_logger
from scripts.validation.scan_common import (
    build_virtual_jobs,
    dataset_id_argument,
    end_option,
    resolve_virtual_dataset,
    start_option,
)
from scripts.validation.utils import output_dir_option, resolve_output_dir

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 32})


def _availability_by_position(
    jobs: list, store: IcechunkStore, append_dim: str
) -> dict[pd.Timestamp, tuple[int, int]]:
    """Map each append-dim position to (present_files, expected_files).

    Reuses VirtualRegionJob.source_file_coords + filter_already_present (which own chunk-key
    resolution) and dedups source files by (position, url) so a file shared across variable
    groups is counted once.
    """
    present_by_file: dict[tuple[pd.Timestamp, str], bool] = {}
    for job in jobs:
        candidates = list(job.source_file_coords())
        missing_ids = {id(c) for c in job.filter_already_present(candidates, store)}
        for coord in candidates:
            key = (coord.out_loc()[append_dim], coord.get_url())
            is_present = id(coord) not in missing_ids
            present_by_file[key] = present_by_file.get(key, False) or is_present

    counts: dict[pd.Timestamp, list[int]] = {}
    for (position, _url), is_present in present_by_file.items():
        bucket = counts.setdefault(position, [0, 0])
        bucket[1] += 1
        bucket[0] += int(is_present)
    return {position: (p, e) for position, (p, e) in counts.items()}


def _plot_availability(
    availability: dict[pd.Timestamp, tuple[int, int]], out_path: Path, dataset_id: str
) -> None:
    positions = sorted(availability)
    fractions = [availability[p][0] / availability[p][1] for p in positions]
    times = np.array(positions, dtype="datetime64[ns]")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, fractions, marker="o", markersize=2, linestyle="-", color="green")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Append-dim position")
    ax.set_ylabel("Fraction of source files present")
    ax.set_title(f"Manifest availability — {dataset_id}", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _write_missing_file(
    incomplete: list[tuple[pd.Timestamp, int, int]], out_path: Path
) -> None:
    lines = [
        "# Incomplete append-dim positions (present/expected source files)",
        "# Retry with backfill --filter-contains <position> to re-ingest missing files.",
        "",
    ]
    positions = [p for p, _, _ in incomplete]
    combined = " ".join(f"--filter-contains {p.isoformat()}" for p in positions)
    lines.append(f"combined-retry-filter: {combined}")
    lines.append("")
    for position, present, expected in incomplete:
        lines.append(f"{position.isoformat()}: {present}/{expected} present")
    out_path.write_text("\n".join(lines))


def scan(
    dataset_id: str = dataset_id_argument,
    start: datetime | None = start_option,
    end: datetime | None = end_option,
    min_fraction: float = typer.Option(
        1.0,
        "--min-fraction",
        help="Required fraction of source files present per position",
    ),
    output_dir: Path | None = output_dir_option,
) -> None:
    """Probe the whole archive for missing source-file refs and report availability."""
    dataset = resolve_virtual_dataset(dataset_id)
    append_dim = dataset.template_config.append_dim
    store = dataset.store_factory.primary_store()
    assert isinstance(store, IcechunkStore)

    log.info(f"Building region jobs for {dataset_id} [{start} .. {end}]")
    jobs = build_virtual_jobs(dataset, end=end, start=start, variables=None)
    log.info(f"Probing manifest across {len(jobs)} region jobs (no decode)")
    availability = _availability_by_position(jobs, store, append_dim)
    assert availability, "No source files generated for the requested window"

    out = resolve_output_dir(dataset.store_factory.primary_url(), output_dir)
    plot_path = out / "manifest_availability.png"
    _plot_availability(availability, plot_path, dataset_id)

    incomplete = sorted(
        (position, present, expected)
        for position, (present, expected) in availability.items()
        if present / expected < min_fraction
    )
    n_positions = len(availability)
    n_present = sum(p for p, _ in availability.values())
    n_expected = sum(e for _, e in availability.values())

    summary = [
        f"# Manifest completeness — {dataset_id}",
        "",
        f"- Positions scanned: {n_positions}",
        f"- Source files present: {n_present}/{n_expected} "
        f"({n_present / n_expected:.2%})",
        f"- Required fraction per position: {min_fraction:.0%}",
        f"- Incomplete positions: {len(incomplete)}",
        "",
        f"![availability]({plot_path.name})",
        "",
    ]
    if incomplete:
        missing_path = out / "missing_source_files.txt"
        _write_missing_file(incomplete, missing_path)
        summary.append(f"Incomplete positions listed in `{missing_path.name}`.")
    (out / "manifest_scan_summary.md").write_text("\n".join(summary))

    log.info(f"Wrote manifest scan to {out}")
    if incomplete:
        log.error(
            f"Manifest incomplete: {len(incomplete)} of {n_positions} positions below "
            f"{min_fraction:.0%} present"
        )
        raise typer.Exit(1)
    log.info(f"Manifest complete: all {n_positions} positions ≥ {min_fraction:.0%}")


if __name__ == "__main__":
    typer.run(scan)
