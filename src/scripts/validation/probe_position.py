"""Name the source files a virtual store is missing at one append-dim position.

`unavailable_timestamps.txt` reports only present/expected counts per position; this
names the specific files behind a count so a gap can be classified as an upstream
absence or an ingestion miss (docs/validation.md §3e).
"""

from typing import Any, cast

import pandas as pd
import typer

from reformatters.common.logging import get_logger
from reformatters.common.virtual_region_job import VirtualRegionJob
from scripts.validation.manifest_scan import (
    coord_is_expected,
    coord_position,
    expected_lead_limits,
    probe_jobs,
)
from scripts.validation.scan_common import build_virtual_jobs, resolve_virtual_dataset
from scripts.validation.utils import load_zarr_dataset, open_icechunk_readonly

log = get_logger(__name__)


def probe_position(dataset_url: str, position: str) -> None:
    """Log every expected source file the store has no reference for at `position`."""
    dataset = resolve_virtual_dataset(
        load_zarr_dataset(dataset_url).attrs["dataset_id"]
    )
    store = open_icechunk_readonly(dataset_url)
    target = pd.Timestamp(position)

    jobs = cast(
        "list[VirtualRegionJob[Any, Any]]",
        build_virtual_jobs(
            dataset,
            start=target,
            end=target + dataset.template_config.append_dim_frequency,
            variables=None,
        ),
    )
    lead_limits = expected_lead_limits(store)

    present = 0
    missing = set()
    for _job, coord_presence in probe_jobs(jobs, store):
        for coord, is_present in coord_presence:
            if coord_position(coord) != target or not coord_is_expected(
                coord, lead_limits
            ):
                continue
            if is_present:
                present += 1
            else:
                missing.add(coord.get_url())

    log.info(f"{target}: {present} present, {len(missing)} missing")
    for url in sorted(missing):
        typer.echo(url)
