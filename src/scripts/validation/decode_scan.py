"""Sampled whole-archive decode-health scan for a virtual dataset.

The offline analog of the operational `CheckVirtualDecodeHealth`, run with wider sampling
across the whole archive instead of just the latest position. It decodes a bounded sample
of present references — across positions, lead times, members, and vertical levels — and
fails if any sampled chunk errors or decodes entirely NaN. This is a sample, not an
exhaustive sweep: a reference that decodes to garbage outside the sample is not caught here
(a literal every-chunk decode is hours; see docs/validation.md).
"""

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import typer
import zarr

from reformatters.common import validation
from reformatters.common.logging import get_logger
from reformatters.common.virtual_region_job import VirtualRegionJob
from scripts.validation.scan_common import (
    build_virtual_jobs,
    dataset_id_argument,
    end_option,
    evenly_spaced_subset,
    resolve_virtual_dataset,
    start_option,
)
from scripts.validation.utils import output_dir_option, resolve_output_dir

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 32})


def scan(
    dataset_id: str = dataset_id_argument,
    start: datetime | None = start_option,
    end: datetime | None = end_option,
    max_samples: int = typer.Option(
        20, "--max-samples", help="Max region jobs (~ positions x var groups) to sample"
    ),
    sampled_leads: int = typer.Option(
        5, help="Lead times to decode per sampled position"
    ),
    sampled_levels: int = typer.Option(
        3, help="Vertical levels to decode per group var"
    ),
    output_dir: Path | None = output_dir_option,
) -> None:
    """Decode a bounded sample of present references across the archive and check health."""
    dataset = resolve_virtual_dataset(dataset_id)
    store = dataset.store_factory.primary_store()
    ds = validation.open_flattened_dataset(store, consolidated=False)

    jobs = build_virtual_jobs(dataset, end=end, start=start, variables=None)
    sampled = evenly_spaced_subset(jobs, max_samples)
    log.info(
        f"Decode-checking {len(sampled)} of {len(jobs)} region jobs "
        f"(sampled_leads={sampled_leads}, sampled_levels={sampled_levels})"
    )

    checker = validation.CheckVirtualDecodeHealth(
        positions="latest",
        sampled_leads=sampled_leads,
        sampled_levels=sampled_levels,
    )
    results = []
    for i, job in enumerate(sampled):
        result = checker(cast(VirtualRegionJob[Any, Any], job), store, ds)  # ty: ignore[invalid-argument-type]
        log.info(f"  [{i + 1}/{len(sampled)}] {'ok' if result.passed else 'FAIL'}")
        results.append(result)

    failures = [r for r in results if not r.passed]
    out = resolve_output_dir(dataset.store_factory.primary_url(), output_dir)
    summary = [
        f"# Decode health (sampled) — {dataset_id}",
        "",
        f"- Region jobs sampled: {len(sampled)} of {len(jobs)}",
        f"- Sampling: latest position per job, {sampled_leads} leads, "
        f"{sampled_levels} levels per group var",
        f"- Failures: {len(failures)}",
        "",
        "Coverage is a sample, not exhaustive: an unsampled reference that decodes to "
        "garbage is not caught here.",
        "",
    ]
    summary.extend(f"- {r.message}" for r in failures)
    (out / "decode_scan_summary.md").write_text("\n".join(summary))
    log.info(f"Wrote decode scan to {out}")

    if failures:
        log.error(f"Decode health failed for {len(failures)} sampled jobs")
        raise typer.Exit(1)
    log.info(f"Decode health passed across {len(sampled)} sampled jobs")


if __name__ == "__main__":
    typer.run(scan)
