"""Whole-archive manifest completeness scan for a virtual dataset.

The offline analog of the operational `CheckVirtualManifestCompleteness`: instead of a
recent window it probes the whole archive for ref existence (no decode), streaming job
by job so peak memory is independent of the scan window length. Two measures:

1. **Per source file** — every expected file's representative ref, the strict
   completeness gate. Lists any incomplete positions for a targeted backfill.
2. **Per variable** — each variable's ref at one present source file per position,
   which catches a variable missing from otherwise-ingested files (e.g. a variable the
   model only started producing partway through the archive).

Entry points: the `availability` command (URL-driven, resolves the registered dataset
from the store's `dataset_id` attribute) and `run-all`, via
`availability.run_manifest_availability`. See docs/validation.md.
"""

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from icechunk.store import IcechunkStore
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common.config_models import DataVar
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.logging import get_logger
from reformatters.common.region_job import SourceFileCoord
from reformatters.common.retry import retry
from reformatters.common.virtual_region_job import VirtualRegionJob, _exists_many
from scripts.validation.scan_common import (
    build_virtual_jobs,
)
from scripts.validation.utils import (
    AvailabilitySeries,
)

log = get_logger(__name__)

_EXISTS_BATCH_SIZE = 20_000
# A whole-archive scan issues enough object-store reads that a transient outage
# (e.g. a connect timeout) is near-certain; ride out a few minutes of trouble rather
# than let one blip kill the whole scan. Backoff grows with attempt in retry(), so
# these attempts span several minutes — 12 was observed too short for a ~90s blip.
_SCAN_MAX_RETRIES = 20


@dataclass
class ManifestScanResult:
    # position -> (present source files, expected source files)
    file_availability: dict[pd.Timestamp, tuple[int, int]]
    # var path -> position -> representative ref present. A position with no present
    # source file for a variable's group is absent (not probed).
    var_availability: dict[str, dict[pd.Timestamp, bool]]


def _position(coord: SourceFileCoord) -> pd.Timestamp:
    position = coord.append_dim_coord
    assert isinstance(position, pd.Timestamp)
    return position


def _probe_job(
    job: VirtualRegionJob[Any, Any], store: IcechunkStore
) -> list[tuple[SourceFileCoord, bool]]:
    """(coord, is_present) for every source file one region job covers."""
    candidates = list(job.source_file_coords())
    # Retried: a whole-archive scan makes enough reads that a transient object store
    # failure is near-certain, and one un-retried probe would kill the run.
    missing = retry(
        lambda: job.filter_already_present(candidates, store),
        max_attempts=_SCAN_MAX_RETRIES,
    )
    missing_ids = {id(c) for c in missing}
    return [(coord, id(coord) not in missing_ids) for coord in candidates]


def _probe_jobs(
    jobs: Sequence[VirtualRegionJob[Any, Any]],
    store: IcechunkStore,
    # 48 network-bound probe workers is the tuned value for the target run host.
    max_workers: int = 48,
) -> Iterator[tuple[VirtualRegionJob[Any, Any], list[tuple[SourceFileCoord, bool]]]]:
    """Probe every job's source files concurrently, yielding (job, [(coord, present)])
    as jobs complete. In-flight work is bounded so a whole-archive scan never holds
    more than a window of jobs' coords at once."""
    job_iter = iter(jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_probe_job, job, store): job
            for job in islice(job_iter, max_workers * 4)
        }
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                for refill in islice(job_iter, 1):
                    futures[pool.submit(_probe_job, refill, store)] = refill
                yield job, future.result()


def _expected_lead_limits(store: IcechunkStore) -> dict[pd.Timestamp, pd.Timedelta]:
    """Per-position maximum expected lead from the store's committed
    expected_forecast_length coordinate; {} when the dataset has none."""
    ds = xr.open_zarr(store, consolidated=False)
    if "expected_forecast_length" not in ds.coords:
        return {}
    lengths = ds["expected_forecast_length"].load()
    (dim,) = lengths.dims
    return dict(
        zip(ds[str(dim)].to_index(), pd.to_timedelta(lengths.values), strict=True)
    )


def _coord_is_expected(
    coord: SourceFileCoord, lead_limits: dict[pd.Timestamp, pd.Timedelta]
) -> bool:
    lead = coord.out_loc().get("lead_time")
    if lead is None or not lead_limits:
        return True
    assert isinstance(lead, pd.Timedelta)
    limit = lead_limits.get(_position(coord))
    # No limit (position not yet committed) or NaT (never written) -> expected.
    return limit is None or pd.isna(limit) or lead <= limit


def _fold_file_availability(
    coord_presence: Sequence[tuple[SourceFileCoord, bool]],
    lead_limits: dict[pd.Timestamp, pd.Timedelta],
    counts: dict[pd.Timestamp, list[int]],
) -> None:
    """Fold one job's probes into per-position [present_files, expected_files] counts.

    Dedups source files by (position, url) so a file shared by several of the job's
    coords is counted once (virtual jobs carry one variable group per region, so a file
    never spans jobs); present if any coord saw it present. Files past a position's
    expected_forecast_length are not expected (e.g. leads beyond a 36-hour-era init).
    """
    present_by_file: dict[tuple[pd.Timestamp, str], bool] = {}
    for coord, is_present in coord_presence:
        if not _coord_is_expected(coord, lead_limits):
            continue
        key = (_position(coord), coord.get_url())
        present_by_file[key] = present_by_file.get(key, False) or is_present

    for (position, _url), is_present in present_by_file.items():
        bucket = counts.setdefault(position, [0, 0])
        bucket[1] += 1
        bucket[0] += int(is_present)


def _has_hour_0_values(var: DataVar[Any]) -> bool:
    """Whether `var` has a ref at lead 0 — mirrors the providers' has_hour_0_values,
    reading only common fields so it works for any virtual dataset."""
    override = var.internal_attrs.hour_0_values_override
    if override is not None:
        return override
    return var.attrs.step_type == "instant"


def _coord_carries(coord: SourceFileCoord, var: DataVar[Any]) -> bool:
    coord_vars = getattr(coord, "data_vars", None)
    return coord_vars is None or any(v.name == var.name for v in coord_vars)


def _sort_coords_for_probe(
    coords: Sequence[SourceFileCoord],
) -> list[tuple[pd.Timedelta | None, SourceFileCoord]]:
    """(lead, coord) pairs in probe-preference order: smallest nonzero lead first,
    lead 0 last (analysis coords have no lead and keep append order). Sorted once per
    position and reused for every variable — a per-variable sort dominates scan CPU."""

    def sort_key(pair: tuple[pd.Timedelta | None, SourceFileCoord]) -> tuple[bool, Any]:
        lead, coord = pair
        if lead is None:
            return (False, coord.append_dim_coord)
        return (lead == pd.Timedelta(0), lead)

    pairs: list[tuple[pd.Timedelta | None, SourceFileCoord]] = []
    for coord in coords:
        lead = coord.out_loc().get("lead_time")
        assert lead is None or isinstance(lead, pd.Timedelta)
        pairs.append((lead, coord))
    return sorted(pairs, key=sort_key)


def _probe_coord_for_var(
    sorted_coords: Sequence[tuple[pd.Timedelta | None, SourceFileCoord]],
    var: DataVar[Any],
) -> SourceFileCoord | None:
    """The present source file to probe `var` at: the first carrier in
    `_sort_coords_for_probe` order.

    Accumulated variables have no ref at lead 0 by design, so if only a lead-0 file is
    present the variable is not probed at this position. Analysis datasets (no
    lead_time in out_loc) probe at any present file.
    """
    skip_lead_0 = not _has_hour_0_values(var)
    for lead, coord in sorted_coords:
        if skip_lead_0 and lead is not None and lead == pd.Timedelta(0):
            continue
        if _coord_carries(coord, var):
            return coord
    return None


@dataclass(frozen=True)
class _VarKeys:
    """Per-variable chunk-key machinery, resolved once: the scan keys each variable at
    thousands of positions and per-probe DataTree/index lookups dominate scan CPU."""

    path: str
    metadata: ArrayV3Metadata
    dims: tuple[str, ...]
    chunks: tuple[int, ...]
    indexes: Mapping[str, pd.Index]
    # Middle chunk of each dim, used for dims out_loc doesn't label (levels).
    middle_chunk: tuple[int, ...]


def _var_keys(
    template_ds: xr.DataTree, group: zarr.Group, var: DataVar[Any]
) -> _VarKeys:
    array = group[var.path]
    assert isinstance(array, zarr.Array)
    assert isinstance(array.metadata, ArrayV3Metadata)
    template_var = template_ds[var.path]
    dims = tuple(str(d) for d in template_var.dims)
    chunks = tuple(template_var.encoding["chunks"])
    middle_chunk = tuple(
        (-(-int(template_var.sizes[dim]) // chunk_size)) // 2
        for dim, chunk_size in zip(dims, chunks, strict=True)
    )
    return _VarKeys(
        path=var.path,
        metadata=array.metadata,
        dims=dims,
        chunks=chunks,
        indexes={str(d): index for d, index in template_var.indexes.items()},
        middle_chunk=middle_chunk,
    )


def _var_chunk_key(keys: _VarKeys, out_loc: Mapping[Any, Any]) -> str:
    """The variable's chunk key at `out_loc`, taking the middle chunk of any unlabeled dim.

    Unlike the write path's chunk resolution (which requires unlabeled dims to be
    single-chunk), a vertical group var's level dim spans many chunks; probing its
    middle chunk mirrors the plots' middle-level sampling.
    """
    index = []
    for dim, chunk_size, middle in zip(
        keys.dims, keys.chunks, keys.middle_chunk, strict=True
    ):
        if dim in out_loc:
            position = keys.indexes[dim].get_loc(out_loc[dim])
            assert isinstance(position, int)
            assert position % chunk_size == 0, (
                f"{keys.path} {dim} label {out_loc[dim]} is not on a chunk boundary"
            )
            index.append(position // chunk_size)
        else:
            index.append(middle)
    encoded = keys.metadata.chunk_key_encoding.encode_chunk_key(tuple(index))
    return f"{keys.path}/{encoded}"


def _var_probes(
    job: VirtualRegionJob[Any, Any],
    coord_presence: Sequence[tuple[SourceFileCoord, bool]],
    group: zarr.Group,
    keys_by_var: dict[str, _VarKeys],
) -> list[tuple[str, pd.Timestamp, str]]:
    """One (var path, position, chunk key) probe per variable per position with a
    present source file: each variable's ref at one present source file per position."""
    by_position: dict[pd.Timestamp, list[SourceFileCoord]] = {}
    for coord, is_present in coord_presence:
        if is_present:
            by_position.setdefault(_position(coord), []).append(coord)
    sorted_by_position = {
        position: _sort_coords_for_probe(coords)
        for position, coords in by_position.items()
    }

    probes: list[tuple[str, pd.Timestamp, str]] = []
    for var in job.data_vars:
        if var.path not in keys_by_var:
            keys_by_var[var.path] = _var_keys(job.template_ds, group, var)
        keys = keys_by_var[var.path]
        for position, sorted_coords in sorted_by_position.items():
            coord = _probe_coord_for_var(sorted_coords, var)
            if coord is None:
                continue
            probes.append((var.path, position, _var_chunk_key(keys, coord.out_loc())))
    return probes


def _flush_var_probes(
    store: IcechunkStore,
    probes: list[tuple[str, pd.Timestamp, str]],
    out: dict[str, dict[pd.Timestamp, bool]],
) -> None:
    keys = [key for _, _, key in probes]
    present = _exists_many(store, keys, max_attempts=_SCAN_MAX_RETRIES)
    for var_path, position, key in probes:
        out.setdefault(var_path, {})[position] = present[key]
    probes.clear()


def scan_manifest(
    dataset: DynamicalDataset[Any, Any],
    store: IcechunkStore,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    variables: list[str] | None = None,
) -> ManifestScanResult:
    """Probe `store`'s manifest per source file and per variable. No decode.

    `dataset` supplies the region-job machinery (expected source files, chunk keys);
    `store` is the archive actually probed, so a staging store is scanned as itself.
    """
    log.info(f"Building region jobs for {dataset.dataset_id} [{start} .. {end}]")
    jobs = cast(
        "list[VirtualRegionJob[Any, Any]]",
        build_virtual_jobs(dataset, end=end, start=start, variables=variables),
    )
    log.info(f"Probing manifest across {len(jobs)} region jobs (no decode)")
    lead_limits = _expected_lead_limits(store)
    group = zarr.open_group(store, mode="r")
    keys_by_var: dict[str, _VarKeys] = {}

    # Fold each job's probes as it completes; only per-position counts, per-var
    # booleans, and at most one _exists_many batch of pending probes stay resident.
    file_counts: dict[pd.Timestamp, list[int]] = {}
    var_availability: dict[str, dict[pd.Timestamp, bool]] = {}
    pending_probes: list[tuple[str, pd.Timestamp, str]] = []
    progress_every = max(1, len(jobs) // 20)
    for i, (job, coord_presence) in enumerate(_probe_jobs(jobs, store), start=1):
        _fold_file_availability(coord_presence, lead_limits, file_counts)
        for var in job.data_vars:
            var_availability.setdefault(var.path, {})
        pending_probes.extend(_var_probes(job, coord_presence, group, keys_by_var))
        if len(pending_probes) >= _EXISTS_BATCH_SIZE:
            _flush_var_probes(store, pending_probes, var_availability)
        if i % progress_every == 0 or i == len(jobs):
            log.info(f"  probed {i}/{len(jobs)} region jobs")
    _flush_var_probes(store, pending_probes, var_availability)

    file_availability = {position: (p, e) for position, (p, e) in file_counts.items()}
    assert file_availability, "No source files generated for the requested window"
    return ManifestScanResult(
        file_availability=file_availability,
        var_availability=var_availability,
    )


def result_availability_series(
    result: ManifestScanResult,
) -> dict[str, AvailabilitySeries]:
    """One AvailabilitySeries per var over every scanned position (NaN = not probed)."""
    all_positions = np.array(sorted(result.file_availability), dtype="datetime64[ns]")
    position_index = pd.DatetimeIndex(all_positions)
    series: dict[str, AvailabilitySeries] = {}
    for var_path, present_by_position in sorted(result.var_availability.items()):
        fraction = np.full(len(all_positions), np.nan)
        for i, position in enumerate(position_index):
            present = present_by_position.get(position)
            if present is not None:
                fraction[i] = 1.0 if present else 0.0
        series[var_path] = AvailabilitySeries(
            positions=all_positions, fraction=fraction
        )
    return series


def write_incomplete_positions_file(
    incomplete: dict[pd.Timestamp, tuple[int, int]], output_dir: Path
) -> str:
    """Write unavailable_timestamps.txt listing incomplete positions. Returns filename."""
    filename = "unavailable_timestamps.txt"
    positions = sorted(incomplete)
    lines = [
        "# Unavailable timestamps (present/expected source files per position)",
        "",
    ]
    for position in positions:
        present, expected = incomplete[position]
        lines.append(f"{position.isoformat()}: {present}/{expected} present")
    (output_dir / filename).write_text("\n".join(lines))
    return filename
