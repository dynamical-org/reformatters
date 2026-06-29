"""Shared helpers for the dataset-aware virtual store scans (manifest + decode).

Unlike the URL-driven plots, these scans need the dataset's own region job to reach the
icechunk manifest (chunk-key resolution + ref-existence probing). Resolving the dataset
from the registry lets them reuse `VirtualRegionJob.source_file_coords` /
`filter_already_present` and the operational validators in `common/validation.py`, so the
offline whole-archive checks and the bounded operational ones share their core logic.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import typer

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.region_job import RegionJob
from reformatters.common.virtual_region_job import VirtualRegionJob

dataset_id_argument = typer.Argument(..., help="Registered dataset id to scan")

start_option = typer.Option(
    None, "--start", help="Restrict the scan to append-dim positions at/after this date"
)
end_option = typer.Option(
    None,
    "--end",
    help="Last append-dim position to scan (default: now). For a post-backfill pass, "
    "pass the backfill's published end so unpublished positions aren't flagged missing.",
)


def resolve_virtual_dataset(dataset_id: str) -> DynamicalDataset[Any, Any]:
    """Look up a registered virtual dataset by id, or raise a CLI error."""
    from reformatters.__main__ import DYNAMICAL_DATASETS  # noqa: PLC0415

    for dataset in DYNAMICAL_DATASETS:
        if dataset.dataset_id == dataset_id:
            break
    else:
        known = ", ".join(d.dataset_id for d in DYNAMICAL_DATASETS)
        raise typer.BadParameter(f"Unknown dataset id {dataset_id!r}. Known: {known}")
    if not issubclass(dataset.region_job_class, VirtualRegionJob):
        raise typer.BadParameter(
            f"{dataset_id!r} is not a virtual dataset; these scans only apply to "
            "virtual (reference) stores."
        )
    return dataset


def build_virtual_jobs(
    dataset: DynamicalDataset[Any, Any],
    *,
    end: datetime | None,
    start: datetime | None,
    variables: list[str] | None,
) -> list[RegionJob[Any, Any]]:
    """Every region job covering [start, end) — the same partitioning a backfill uses."""
    append_dim_end = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    template_ds = dataset._get_template(append_dim_end)  # noqa: SLF001
    return list(
        dataset.region_job_class.get_jobs(
            tmp_store=dataset._tmp_store(),  # noqa: SLF001
            template_ds=template_ds,
            append_dim=dataset.template_config.append_dim,
            all_data_vars=dataset.template_config.data_vars,
            reformat_job_name="validation-scan",
            filter_start=pd.Timestamp(start) if start is not None else None,
            filter_variable_names=variables,
        )
    )


def evenly_spaced_subset(items: list[Any], n: int) -> list[Any]:
    """Up to `n` evenly spaced items (first, last, and spread between)."""
    if n <= 0 or len(items) <= n:
        return items
    idxs = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idxs]
