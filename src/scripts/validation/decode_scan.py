"""Sampled whole-archive decode-health scan for a virtual dataset.

The offline analog of the operational `CheckVirtualDecodeHealth`, run with wider sampling
across the whole archive instead of just the latest position. It decodes a bounded sample
of present references — across positions, lead times, members, and vertical levels — and
fails if any sampled chunk errors or decodes entirely NaN. This is a sample, not an
exhaustive sweep: a reference that decodes to garbage outside the sample is not caught here
(a literal every-chunk decode is hours; see docs/validation.md).

Entry points: the `decode-scan` command (URL-driven, resolves the registered dataset from
the store's `dataset_id` attribute) and `run-all`, via `run_decode_scan`. Both accept
`--checkpoint-dir`, which records each sampled region job as it finishes so an
interrupted scan resumes. See docs/validation.md.
"""

import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pandas as pd
import typer
import zarr

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.iterating import digest
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.virtual_region_job import VirtualRegionJob, _exists_many
from scripts.validation.availability import build_run_context
from scripts.validation.manifest_scan import _var_chunk_keys, _var_keys, _VarKeys
from scripts.validation.scan_common import (
    build_virtual_jobs,
    evenly_spaced_subset,
    read_checkpoint,
    resolve_scan_window,
    write_checkpoint,
)
from scripts.validation.utils import (
    RunContext,
    checkpoint_dir_option,
    end_date_option,
    output_dir_option,
    start_date_option,
    variables_option,
)

log = get_logger(__name__)

MAX_SAMPLED_REGIONS = 20
SAMPLED_LEADS = 5
JOB_CONCURRENCY = 4


def _decode_checker(
    dataset: DynamicalDataset[Any, Any],
    reference_exists: Callable[[str, Mapping[str, Any]], bool],
) -> validation.CheckVirtualDecodeHealth:
    configured = next(
        (
            validator
            for validator in dataset.validators()
            if isinstance(validator, validation.CheckVirtualDecodeHealth)
        ),
        validation.CheckVirtualDecodeHealth(),
    )
    return configured.model_copy(
        update={
            "positions": 1,
            "sampled_leads": SAMPLED_LEADS,
            "reference_exists": reference_exists,
        }
    )


def _checkpoint_key(
    dataset_id: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
    variables: list[str],
    max_samples: int,
    checker: validation.CheckVirtualDecodeHealth,
) -> str:
    """Namespace under which a sampled job's result may be reused: the scan plan (window,
    variable filter, region sample size), so a changed extent never reuses a job recorded
    while its tail was still being filled, and the checker configuration that decides a
    job's outcome (lead, level and position sampling, all-NaN allowances). The job's own
    region and variables are in its file name."""
    return digest(
        [
            json.dumps(
                {
                    "version": 2,
                    "dataset_id": dataset_id,
                    "start": str(start),
                    "end": str(end),
                    "variables": sorted(variables),
                    "max_samples": max_samples,
                    "checker": checker.model_dump(
                        mode="json", exclude={"reference_exists", "max_workers"}
                    ),
                },
                sort_keys=True,
            )
        ]
    )


def _job_checkpoint_path(
    checkpoint_dir: Path, dataset_id: str, key: str, job: RegionJob[Any, Any]
) -> Path:
    """Where one sampled region job's outcome is recorded — one file per job, since a job
    is the unit of work a resumed scan skips. The dataset id and key let one directory
    hold both scans' files for several datasets."""
    variables = digest(sorted(var.path for var in job.data_vars))
    job_id = f"{job.region.start}-{job.region.stop}-{variables}"
    return checkpoint_dir / f"{dataset_id}_decode_{key}_{job_id}.json"


def _write_job_checkpoint(path: Path, result: validation.ValidationResult) -> None:
    write_checkpoint(path, result.model_dump(mode="json"))


def _read_job_checkpoint(path: Path) -> validation.ValidationResult:
    return validation.ValidationResult.model_validate(read_checkpoint(path))


def run_decode_scan(ctx: RunContext, max_samples: int = MAX_SAMPLED_REGIONS) -> None:
    """Decode a bounded sample of present references and record health on ctx.

    With `ctx.checkpoint_dir` each sampled region job's outcome is written there as it
    finishes and reused on a later call, so an interrupted scan decodes only the jobs it
    did not reach.
    """
    assert ctx.is_virtual, "decode scan reads refs from a virtual store's manifest"
    dataset, store, start, end = resolve_scan_window(ctx)
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
        keys = _var_chunk_keys(keys_by_var[var_path], out_loc)
        return any(_exists_many(store, keys).values())

    jobs = build_virtual_jobs(dataset, end=end, start=start, variables=ctx.variables)
    # Sample evenly over append-dim regions, keeping every var-group job at each sampled
    # region — sampling the raw job list would stride over (region x var group) and could
    # systematically skip whole variable groups when a dataset sets max_vars_per_job.
    regions = sorted({job.region.start for job in jobs})
    sampled_regions = set(evenly_spaced_subset(regions, max_samples))
    sampled = [job for job in jobs if job.region.start in sampled_regions]
    checker = _decode_checker(dataset, reference_exists)
    log.info(
        f"Decode-checking {len(sampled)} of {len(jobs)} region jobs across "
        f"{len(sampled_regions)} of {len(regions)} regions "
        f"(sampled_leads={checker.sampled_leads}, sampled_levels={checker.sampled_levels})"
    )

    key = _checkpoint_key(
        dataset.dataset_id, start, end, ctx.variables, max_samples, checker
    )

    def checkpoint_path(job: RegionJob[Any, Any]) -> Path | None:
        if ctx.checkpoint_dir is None:
            return None
        return _job_checkpoint_path(ctx.checkpoint_dir, dataset.dataset_id, key, job)

    def check(job: RegionJob[Any, Any]) -> validation.ValidationResult:
        result = checker.check(
            validation.ValidationContext(
                store=store,
                ds=ds,
                append_dim=dataset.template_config.append_dim,
                data_vars=dataset.template_config.data_vars,
                region_job=cast("VirtualRegionJob[Any, Any]", job),
            )
        )
        path = checkpoint_path(job)
        if path is not None:
            _write_job_checkpoint(path, result)
        return result

    checkpointed = {
        index: _read_job_checkpoint(path)
        for index, job in enumerate(sampled)
        if (path := checkpoint_path(job)) is not None and path.exists()
    }
    pending = [job for index, job in enumerate(sampled) if index not in checkpointed]
    if checkpointed:
        log.info(
            f"Reusing {len(checkpointed)} checkpointed region jobs, "
            f"decoding the remaining {len(pending)}"
        )

    failures = []
    decoded_refs = 0
    # A job's decodes are network-latency-bound and parallelize only across its own
    # source files, so a few jobs run concurrently to fill the idle time.
    with ThreadPoolExecutor(max_workers=JOB_CONCURRENCY) as pool:
        decoded = iter(pool.map(check, pending))
        for i in range(len(sampled)):
            result = checkpointed[i] if i in checkpointed else next(decoded)
            log.info(f"  [{i + 1}/{len(sampled)}] {'ok' if result.passed else 'FAIL'}")
            decoded_refs += result.checked_count or 0
            if not result.passed:
                failures.append(result.message)

    ctx.decode_sample_desc = (
        f"{len(sampled_regions)} of {len(regions)} append-dim regions, "
        f"{checker.sampled_leads} leads and up to {checker.sampled_levels} "
        "present levels per group variable"
    )
    ctx.decode_checked_count = decoded_refs
    ctx.decode_failures = failures
    if failures:
        log.error(f"Decode health failed for {len(failures)} sampled jobs")
    else:
        log.info(f"Decode health passed across {len(sampled)} sampled jobs")


def decode_summary_lines(ctx: RunContext) -> list[str]:
    assert ctx.decode_sample_desc is not None
    assert ctx.decode_failures is not None
    if ctx.decode_failures:
        return [
            f"Decode health failures, sampled across {ctx.decode_sample_desc}:",
            "",
            *(f"- FAIL: {message}" for message in ctx.decode_failures),
        ]
    return [
        (
            f"{ctx.decode_checked_count} references decoded successfully, "
            f"sampled across {ctx.decode_sample_desc}."
        )
    ]


def decode_scan(
    dataset_url: str,
    variables: list[str] | None = variables_option,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
    output_dir: Path | None = output_dir_option,
    checkpoint_dir: Path | None = checkpoint_dir_option,
    max_samples: int = typer.Option(
        MAX_SAMPLED_REGIONS,
        "--max-samples",
        help="Max append-dim regions to decode-check",
    ),
) -> None:
    """Decode a bounded sample of present references across the archive and check health."""
    ctx = build_run_context(
        dataset_url,
        variables,
        start_date,
        end_date,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
    )
    run_decode_scan(ctx, max_samples=max_samples)
    (ctx.output_dir / "decode_scan_summary.md").write_text(
        "\n".join(["# Decode health", "", *decode_summary_lines(ctx)])
    )
    if ctx.decode_failures:
        raise typer.Exit(1)
