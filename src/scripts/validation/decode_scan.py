"""Sampled whole-archive decode-health scan for a virtual dataset.

The offline analog of the operational `CheckVirtualDecodeHealth`, run with wider sampling
across the whole archive instead of just the latest position. It decodes a bounded sample
of present references — across positions, lead times, members, and vertical levels — and
fails if any sampled chunk errors or decodes entirely NaN. This is a sample, not an
exhaustive sweep: a reference that decodes to garbage outside the sample is not caught here
(a literal every-chunk decode is hours; see docs/validation.md).

Entry points: the `decode-scan` command (URL-driven, resolves the registered dataset from
the store's `dataset_id` attribute) and `run-all`, via `run_decode_scan`.
"""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pandas as pd
import typer
import zarr

from reformatters.common import validation
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.virtual_region_job import VirtualRegionJob, _exists_many
from scripts.validation.availability import build_run_context
from scripts.validation.manifest_scan import _var_chunk_key, _var_keys, _VarKeys
from scripts.validation.scan_common import (
    build_virtual_jobs,
    evenly_spaced_subset,
    resolve_virtual_dataset,
)
from scripts.validation.utils import (
    RunContext,
    end_date_option,
    open_icechunk_readonly,
    output_dir_option,
    start_date_option,
    variables_option,
)

log = get_logger(__name__)

zarr.config.set({"async.concurrency": 32})

MAX_SAMPLED_REGIONS = 20
SAMPLED_LEADS = 5
SAMPLED_LEVELS = 3
JOB_CONCURRENCY = 4


def run_decode_scan(ctx: RunContext, max_samples: int = MAX_SAMPLED_REGIONS) -> None:
    """Decode a bounded sample of present references and record health on ctx."""
    assert ctx.is_virtual, "decode scan reads refs from a virtual store's manifest"
    dataset = resolve_virtual_dataset(ctx.validation_ds.attrs["dataset_id"])
    append_dim = dataset.template_config.append_dim
    # End is exclusive; extend one step past the committed extent so the newest
    # position is sampled without flagging not-yet-published ones.
    end = (
        pd.Timestamp(ctx.validation_ds[append_dim].max().item())
        + dataset.template_config.append_dim_frequency
    )
    start = pd.Timestamp(ctx.start_date) if ctx.start_date else None

    store = open_icechunk_readonly(ctx.validation_url)
    ds = validation.open_flattened_dataset(store, consolidated=False)

    template_ds = dataset.template_config.get_template(end)
    group = zarr.open_group(store, mode="r")
    var_by_path = {v.path: v for v in dataset.template_config.data_vars}
    # Pre-build all _VarKeys single-threaded so the oracle only READS the cache (decode()
    # runs in a ThreadPoolExecutor; concurrent cache writes would race).
    keys_by_var: dict[str, _VarKeys] = {
        path: _var_keys(template_ds, group, var) for path, var in var_by_path.items()
    }

    def reference_exists(var_path: str, out_loc: Mapping[str, Any]) -> bool:
        key = _var_chunk_key(keys_by_var[var_path], out_loc)
        return _exists_many(store, [key])[key]

    jobs = build_virtual_jobs(dataset, end=end, start=start, variables=ctx.variables)
    # Sample evenly over append-dim regions, keeping every var-group job at each sampled
    # region — sampling the raw job list would stride over (region x var group) and could
    # systematically skip whole variable groups when a dataset sets max_vars_per_job.
    regions = sorted({job.region.start for job in jobs})
    sampled_regions = set(evenly_spaced_subset(regions, max_samples))
    sampled = [job for job in jobs if job.region.start in sampled_regions]
    log.info(
        f"Decode-checking {len(sampled)} of {len(jobs)} region jobs across "
        f"{len(sampled_regions)} of {len(regions)} regions "
        f"(sampled_leads={SAMPLED_LEADS}, sampled_levels={SAMPLED_LEVELS})"
    )

    checker = validation.CheckVirtualDecodeHealth(
        positions="latest",
        sampled_leads=SAMPLED_LEADS,
        sampled_levels=SAMPLED_LEVELS,
        reference_exists=reference_exists,
    )

    def check(job: RegionJob[Any, Any]) -> validation.ValidationResult:
        return checker(cast("VirtualRegionJob[Any, Any]", job), store, ds)

    failures = []
    # A job's decodes are network-latency-bound and parallelize only across its own
    # source files, so a few jobs run concurrently to fill the idle time.
    with ThreadPoolExecutor(max_workers=JOB_CONCURRENCY) as pool:
        for i, result in enumerate(pool.map(check, sampled)):
            log.info(f"  [{i + 1}/{len(sampled)}] {'ok' if result.passed else 'FAIL'}")
            if not result.passed:
                failures.append(result.message)

    ctx.decode_note = (
        "Decode health decodes a bounded sample of references that EXIST and checks "
        f"they read as real data: {len(sampled_regions)} of {len(regions)} append-dim "
        f"regions, the latest present position per region, {SAMPLED_LEADS} leads and "
        f"{SAMPLED_LEVELS} levels per group variable, all members. References that "
        "don't exist are reported by the availability check, not here. An unsampled "
        "reference that decodes to garbage is not caught here."
    )
    ctx.decode_failures = failures
    if failures:
        log.error(f"Decode health failed for {len(failures)} sampled jobs")
    else:
        log.info(f"Decode health passed across {len(sampled)} sampled jobs")


def decode_summary_lines(ctx: RunContext) -> list[str]:
    assert ctx.decode_note is not None
    assert ctx.decode_failures is not None
    lines = [ctx.decode_note, ""]
    if ctx.decode_failures:
        lines.extend(f"- FAIL: {message}" for message in ctx.decode_failures)
    else:
        lines.append("All sampled references decoded successfully.")
    return lines


def decode_scan(
    dataset_url: str,
    variables: list[str] | None = variables_option,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
    output_dir: Path | None = output_dir_option,
    max_samples: int = typer.Option(
        MAX_SAMPLED_REGIONS,
        "--max-samples",
        help="Max append-dim regions to decode-check",
    ),
) -> None:
    """Decode a bounded sample of present references across the archive and check health."""
    ctx = build_run_context(
        dataset_url, variables, start_date, end_date, output_dir=output_dir
    )
    run_decode_scan(ctx, max_samples=max_samples)
    (ctx.output_dir / "decode_scan_summary.md").write_text(
        "\n".join(["# Decode health (sampled)", "", *decode_summary_lines(ctx)])
    )
    if ctx.decode_failures:
        raise typer.Exit(1)
