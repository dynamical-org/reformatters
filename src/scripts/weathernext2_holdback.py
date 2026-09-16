#!/usr/bin/env python3
"""Audit, remove, and publish WeatherNext 2 holdback refs."""

import asyncio
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import batched, groupby, product
from math import ceil
from pathlib import Path
from typing import Annotated, Any, cast

import icechunk
import numpy as np
import pandas as pd
import typer
import zarr
from icechunk.store import IcechunkStore
from numpy.typing import NDArray
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.logging import get_logger
from reformatters.common.virtual_region_job import _exists_many
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT,
    OPERATIONAL_ROOT_MANIFEST_INIT_SPLIT,
    PUBLICATION_HOLDBACK,
)
from scripts.validation.scan_common import resolve_virtual_dataset
from scripts.validation.utils import _icechunk_storage

_BATCH_SIZE = 20_000
_OPERATIONAL_DATASET_ID = "google-weathernext2-forecast-operational-virtual"
_PURGE_BRANCH = "holdback-purge"

log = get_logger(__name__)
app = typer.Typer(
    help="Audit, remove, and publish WeatherNext 2 holdback refs.",
    pretty_exceptions_show_locals=False,
)


@dataclass(frozen=True)
class ForbiddenChunk:
    var_path: str
    init_time: pd.Timestamp
    lead_time: pd.Timedelta
    key: str


@dataclass
class StepCounts:
    present: int = 0
    total: int = 0


@dataclass(frozen=True)
class HoldbackAudit:
    total_keys: int
    present_keys: int
    counts: dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts]
    report_path: Path

    @property
    def present_steps(self) -> set[tuple[pd.Timestamp, pd.Timedelta]]:
        return {
            (init_time, lead_time)
            for (_var_path, init_time, lead_time), counts in self.counts.items()
            if counts.present
        }

    @property
    def present_inits(self) -> set[pd.Timestamp]:
        return {init_time for init_time, _lead_time in self.present_steps}

    @property
    def max_present_valid_time(self) -> pd.Timestamp | None:
        if not self.present_steps:
            return None
        return max(init_time + lead_time for init_time, lead_time in self.present_steps)

    @property
    def last_age_out(self) -> pd.Timestamp | None:
        if self.max_present_valid_time is None:
            return None
        return self.max_present_valid_time + PUBLICATION_HOLDBACK


def _arrays(group: zarr.Group) -> Iterator[zarr.Array]:
    for _name, array in group.arrays():
        yield array
    for _name, subgroup in group.groups():
        yield from _arrays(subgroup)


def _coordinate_values(
    root: zarr.Group, group_path: str
) -> tuple[pd.DatetimeIndex, pd.TimedeltaIndex]:
    group = root if not group_path else root[group_path]
    assert isinstance(group, zarr.Group)
    init_array = group["init_time"]
    lead_array = group["lead_time"]
    assert isinstance(init_array, zarr.Array)
    assert isinstance(lead_array, zarr.Array)
    assert init_array.ndim == lead_array.ndim == 1
    assert init_array.dtype == np.dtype("int64")
    assert lead_array.dtype == np.dtype("float64")
    assert str(init_array.attrs["units"]).startswith("seconds since 1970-01-01")
    assert lead_array.attrs["units"] == "seconds", lead_array.path
    init_values = cast("NDArray[np.int64]", init_array[:])
    lead_values = cast("NDArray[np.float64]", lead_array[:])
    init_seconds = [int(value) for value in init_values]
    lead_seconds = [float(value) for value in lead_values]
    return (
        pd.DatetimeIndex(pd.to_datetime(init_seconds, unit="s")),
        pd.TimedeltaIndex(pd.to_timedelta(lead_seconds, unit="s")),
    )


def _chunk_grid_indexes(
    array: zarr.Array,
    dims: tuple[str | None, ...],
    init_index: int,
    lead_index: int,
) -> Iterator[tuple[int, ...]]:
    chunk_ranges: list[Sequence[int]] = []
    for dim, size, chunk_size in zip(dims, array.shape, array.chunks, strict=True):
        if dim == "init_time":
            chunk_ranges.append((init_index,))
        elif dim == "lead_time":
            chunk_ranges.append((lead_index,))
        else:
            chunk_ranges.append(range(ceil(size / chunk_size)))
    yield from product(*chunk_ranges)


def forbidden_chunk_keys(
    group: zarr.Group, cutoff: pd.Timestamp
) -> Iterator[ForbiddenChunk]:
    assert cutoff.tz is None, "cutoff must be normalized to naive UTC"
    coordinates: dict[str, tuple[pd.DatetimeIndex, pd.TimedeltaIndex]] = {}
    found_data_array = False
    for array in _arrays(group):
        metadata = array.metadata
        assert isinstance(metadata, ArrayV3Metadata)
        dimension_names = metadata.dimension_names
        if dimension_names is None:
            assert array.shape == (), array.path
            continue
        dims = tuple(dimension_names)
        if not {"init_time", "lead_time", "y", "x"} <= set(dims):
            continue
        found_data_array = True
        if metadata.shards is not None:
            raise NotImplementedError(f"sharded arrays are unsupported: {array.path}")
        assert "ensemble_member" in dims, array.path
        init_dim = dims.index("init_time")
        lead_dim = dims.index("lead_time")
        chunks = tuple(array.chunks)
        assert chunks[init_dim] == chunks[lead_dim] == 1, array.path

        group_path = array.path.rpartition("/")[0]
        if group_path not in coordinates:
            coordinates[group_path] = _coordinate_values(group, group_path)
        init_times, lead_times = coordinates[group_path]
        assert len(init_times) == array.shape[init_dim]
        assert len(lead_times) == array.shape[lead_dim]
        assert len(lead_times) > 0
        max_lead = lead_times.max()

        for init_index, init_time in enumerate(init_times):
            if init_time + max_lead <= cutoff:
                continue
            for lead_index, lead_time in enumerate(lead_times):
                if init_time + lead_time <= cutoff:
                    continue
                for chunk_index in _chunk_grid_indexes(
                    array, dims, init_index, lead_index
                ):
                    encoded = metadata.chunk_key_encoding.encode_chunk_key(chunk_index)
                    yield ForbiddenChunk(
                        var_path=array.path,
                        init_time=init_time,
                        lead_time=lead_time,
                        key=f"{array.path}/{encoded}",
                    )
    assert found_data_array, "no WeatherNext 2 data arrays found"


async def _delete_many(store: IcechunkStore, keys: Sequence[str]) -> None:
    await asyncio.gather(*(store.delete(key) for key in keys))


def _probe_chunks(
    store: IcechunkStore,
    chunks: Iterator[ForbiddenChunk],
    *,
    delete_present: bool = False,
) -> tuple[int, int, dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts]]:
    total_keys = 0
    present_keys = 0
    counts: dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts] = {}
    for chunk_batch in batched(chunks, _BATCH_SIZE, strict=False):
        keys = [chunk.key for chunk in chunk_batch]
        presence = _exists_many(store, keys)
        present_batch = [chunk.key for chunk in chunk_batch if presence[chunk.key]]
        if delete_present and present_batch:
            asyncio.run(_delete_many(store, present_batch))
        for chunk in chunk_batch:
            step = counts.setdefault(
                (chunk.var_path, chunk.init_time, chunk.lead_time), StepCounts()
            )
            step.total += 1
            step.present += int(presence[chunk.key])
        total_keys += len(chunk_batch)
        present_keys += len(present_batch)
        log.info(f"Probed {total_keys:,} forbidden keys; {present_keys:,} present")
    return total_keys, present_keys, counts


def _probe(
    store: IcechunkStore,
    group: zarr.Group,
    cutoff: pd.Timestamp,
) -> tuple[int, int, dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts]]:
    return _probe_chunks(store, forbidden_chunk_keys(group, cutoff))


def _merge_counts(
    target: dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts],
    source: dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts],
) -> None:
    for key, source_counts in source.items():
        target_counts = target.setdefault(key, StepCounts())
        target_counts.present += source_counts.present
        target_counts.total += source_counts.total


def _manifest_window(
    chunk: ForbiddenChunk,
    init_indexes: dict[str, dict[pd.Timestamp, int]],
    root_split: int,
    pressure_split: int,
) -> tuple[str, int]:
    if "/" not in chunk.var_path:
        group_path = ""
        split = root_split
    else:
        assert chunk.var_path.startswith("pressure_level/"), chunk.var_path
        group_path = "pressure_level"
        split = pressure_split
    return chunk.var_path, init_indexes[group_path][chunk.init_time] // split


def _snapshot_descends_from(repo: icechunk.Repository, tip: str, ancestor: str) -> bool:
    return any(snapshot.id == ancestor for snapshot in repo.ancestry(snapshot_id=tip))


def _operational_dataset(dataset_id: str) -> DynamicalDataset[Any, Any]:
    if dataset_id != _OPERATIONAL_DATASET_ID:
        raise typer.BadParameter(
            f"destructive commands only support {_OPERATIONAL_DATASET_ID!r}"
        )
    return resolve_virtual_dataset(dataset_id)


def _default_cutoff() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None) - PUBLICATION_HOLDBACK


def _parse_cutoff(value: str | None) -> pd.Timestamp:
    if value is None:
        return _default_cutoff()
    cutoff = pd.Timestamp(value)
    if cutoff.tz is None:
        raise typer.BadParameter("--cutoff must include a UTC offset")
    return cutoff.tz_convert("UTC").tz_localize(None)


def _utc_iso(value: pd.Timestamp | None) -> str:
    if value is None:
        return "none"
    assert value.tz is None
    return value.tz_localize("UTC").isoformat()


def _default_output_dir() -> Path:
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H-%M")
    return Path("data/output") / f"holdback_audit_{timestamp}"


def _write_report(
    *,
    store_label: str,
    snapshot: icechunk.SnapshotInfo,
    cutoff: pd.Timestamp,
    total_keys: int,
    present_keys: int,
    counts: dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts],
    output_dir: Path,
) -> HoldbackAudit:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = HoldbackAudit(
        total_keys=total_keys,
        present_keys=present_keys,
        counts=counts,
        report_path=output_dir / "holdback_audit.md",
    )
    header = [
        "# WeatherNext 2 publication holdback audit",
        "",
        f"- Store: `{store_label}`",
        f"- Snapshot: `{snapshot.id}` ({snapshot.written_at.isoformat()})",
        f"- Cutoff: `{_utc_iso(cutoff)}`",
        f"- Total keys probed: {total_keys:,}",
        f"- Keys present: {present_keys:,}",
        f"- Steps with any present ref: {len(audit.present_steps):,}",
        f"- Inits with any present ref: {len(audit.present_inits):,}",
        f"- Maximum present valid time: `{_utc_iso(audit.max_present_valid_time)}`",
        f"- Last age-out: `{_utc_iso(audit.last_age_out)}`",
    ]
    for line in header[2:]:
        log.info(line.removeprefix("- "))

    rows = []
    for (var_path, init_time, lead_time), step in sorted(
        counts.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])
    ):
        if not step.present:
            continue
        rows.append(
            "| "
            f"{_utc_iso(init_time)} | {lead_time} | "
            f"{_utc_iso(init_time + lead_time)} | `{var_path}` | "
            f"{step.present:,}/{step.total:,} |"
        )
    if rows:
        detail = [
            "",
            "## Present forbidden refs",
            "",
            "| Init time | Lead time | Valid time | Variable | Present/total |",
            "| --- | --- | --- | --- | ---: |",
            *rows,
        ]
    else:
        detail = ["", "No forbidden refs are present."]
    audit.report_path.write_text("\n".join([*header, *detail, ""]), encoding="utf-8")
    log.info(f"Report: {audit.report_path}")
    return audit


def _run_audit(
    repo: icechunk.Repository,
    store_label: str,
    cutoff: pd.Timestamp,
    snapshot_id: str | None,
    branch: str | None,
    output_dir: Path,
) -> HoldbackAudit:
    assert snapshot_id is None or branch is None
    if snapshot_id is None:
        snapshot_id = repo.lookup_branch(branch or "main")
    snapshot = repo.lookup_snapshot(snapshot_id)
    log.info(f"Snapshot: {snapshot.id} ({snapshot.written_at.isoformat()})")
    log.info(f"Cutoff: {_utc_iso(cutoff)}")
    store = repo.readonly_session(snapshot_id=snapshot.id).store
    group = zarr.open_group(store, mode="r")
    total_keys, present_keys, counts = _probe(store, group, cutoff)
    return _write_report(
        store_label=store_label,
        snapshot=snapshot,
        cutoff=cutoff,
        total_keys=total_keys,
        present_keys=present_keys,
        counts=counts,
        output_dir=output_dir,
    )


def _purge_windows(
    repo: icechunk.Repository,
    group: zarr.Group,
    cutoff: pd.Timestamp,
    root_split: int,
    pressure_split: int,
) -> tuple[
    int,
    int,
    dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts],
    int,
]:
    init_indexes = {}
    for group_path in ("", "pressure_level"):
        init_times, _lead_times = _coordinate_values(group, group_path)
        init_indexes[group_path] = {
            init_time: index for index, init_time in enumerate(init_times)
        }
    windowed_chunks = groupby(
        forbidden_chunk_keys(group, cutoff),
        key=lambda chunk: _manifest_window(
            chunk, init_indexes, root_split, pressure_split
        ),
    )
    total_keys = 0
    present_keys = 0
    counts: dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts] = {}
    groups_committed = 0
    for (var_path, _window), window_chunks in windowed_chunks:
        started = time.monotonic()
        session = repo.writable_session(_PURGE_BRANCH)
        window_total, window_present, window_counts = _probe_chunks(
            session.store, window_chunks, delete_present=True
        )
        total_keys += window_total
        present_keys += window_present
        _merge_counts(counts, window_counts)
        init_times = [key[1] for key in window_counts]
        assert init_times
        first_init = min(init_times)
        last_init = max(init_times)
        if not window_present:
            log.info(
                f"Skipped {var_path} inits {_utc_iso(first_init)}.."
                f"{_utc_iso(last_init)}; all {window_total:,} refs absent"
            )
            continue
        snapshot_id = session.commit(
            f"Remove {window_present} refs with valid time after "
            f"{cutoff:%Y-%m-%dT%H:%M}Z from {var_path} inits "
            f"{first_init.isoformat()}..{last_init.isoformat()}"
        )
        groups_committed += 1
        log.info(
            f"Committed {snapshot_id} for {var_path} inits "
            f"{_utc_iso(first_init)}..{_utc_iso(last_init)}: "
            f"{window_present:,} refs in {time.monotonic() - started:.1f}s"
        )
    return total_keys, present_keys, counts, groups_committed


def _run_delete(
    repo: icechunk.Repository,
    store_label: str,
    cutoff: pd.Timestamp,
    output_dir: Path,
    *,
    force: bool,
    root_split: int = OPERATIONAL_ROOT_MANIFEST_INIT_SPLIT,
    pressure_split: int = OPERATIONAL_PRESSURE_MANIFEST_INIT_SPLIT,
) -> HoldbackAudit:
    assert root_split > 0
    assert pressure_split > 0
    main_start = repo.lookup_branch("main")
    branch_tip: str | None = None
    if _PURGE_BRANCH in repo.list_branches():
        branch_tip = repo.lookup_branch(_PURGE_BRANCH)
        if not _snapshot_descends_from(repo, branch_tip, main_start):
            raise typer.BadParameter(
                f"{_PURGE_BRANCH} does not descend from current main {main_start}"
            )

    if not force:
        read_snapshot = branch_tip or main_start
        snapshot = repo.lookup_snapshot(read_snapshot)
        store = repo.readonly_session(snapshot_id=read_snapshot).store
        group = zarr.open_group(store, mode="r")
        total_keys, present_keys, counts = _probe(store, group, cutoff)
        audit = _write_report(
            store_label=store_label,
            snapshot=snapshot,
            cutoff=cutoff,
            total_keys=total_keys,
            present_keys=present_keys,
            counts=counts,
            output_dir=output_dir,
        )
        log.info(
            f"Would delete {present_keys:,} refs across "
            f"{len(audit.present_steps):,} steps"
        )
        log.info(f"Main start snapshot: {main_start}")
        log.info(f"Holdback branch tip: {branch_tip or 'not created (dry run)'}")
        log.info("Groups committed: 0")
        log.info("Refs deleted: 0")
        log.info(f"Cutoff: {_utc_iso(cutoff)}")
        return audit

    if branch_tip is None:
        repo.create_branch(_PURGE_BRANCH, main_start)
        branch_tip = main_start

    snapshot = repo.lookup_snapshot(branch_tip)
    enumeration_store = repo.readonly_session(snapshot_id=branch_tip).store
    group = zarr.open_group(enumeration_store, mode="r")
    total_keys, present_keys, counts, groups_committed = _purge_windows(
        repo, group, cutoff, root_split, pressure_split
    )

    audit = _write_report(
        store_label=store_label,
        snapshot=snapshot,
        cutoff=cutoff,
        total_keys=total_keys,
        present_keys=present_keys,
        counts=counts,
        output_dir=output_dir,
    )
    branch_tip = repo.lookup_branch(_PURGE_BRANCH)
    fresh_store = repo.readonly_session(snapshot_id=branch_tip).store
    fresh_group = zarr.open_group(fresh_store, mode="r")
    verified_total, verified_present, _counts = _probe(fresh_store, fresh_group, cutoff)
    assert verified_total == total_keys
    assert verified_present == 0, (
        f"{verified_present} forbidden refs remain on {_PURGE_BRANCH} {branch_tip}"
    )
    log.info(
        f"Verified all {verified_total:,} forbidden keys absent on {_PURGE_BRANCH}"
    )
    log.info(f"Main start snapshot: {main_start}")
    log.info(f"Holdback branch tip: {branch_tip}")
    log.info(f"Groups committed: {groups_committed:,}")
    log.info(f"Refs deleted: {present_keys:,}")
    log.info(f"Cutoff: {_utc_iso(cutoff)}")
    return audit


def _run_publish(
    repo: icechunk.Repository,
    store_label: str,
    from_snapshot: str,
    cutoff: pd.Timestamp,
    output_dir: Path,
    *,
    force: bool,
) -> HoldbackAudit:
    current_main = repo.lookup_branch("main")
    if current_main != from_snapshot:
        raise typer.BadParameter(
            f"main is at {current_main}, not --from-snapshot {from_snapshot}"
        )
    if _PURGE_BRANCH not in repo.list_branches():
        raise typer.BadParameter(f"Branch {_PURGE_BRANCH!r} does not exist")
    branch_tip = repo.lookup_branch(_PURGE_BRANCH)
    if not _snapshot_descends_from(repo, branch_tip, from_snapshot):
        raise typer.BadParameter(
            f"{_PURGE_BRANCH} tip {branch_tip} does not descend from "
            f"--from-snapshot {from_snapshot}"
        )
    result = _run_audit(
        repo,
        store_label,
        cutoff,
        branch_tip,
        None,
        output_dir,
    )
    if result.present_keys:
        raise typer.Exit(1)
    if not force:
        log.info(
            f"Would reset main from {from_snapshot} to {branch_tip} and delete "
            f"{_PURGE_BRANCH}"
        )
        return result
    repo.reset_branch("main", branch_tip, from_snapshot_id=from_snapshot)
    repo.delete_branch(_PURGE_BRANCH)
    log.info(f"Reset main from {from_snapshot} to {branch_tip}")
    log.info(f"Deleted branch {_PURGE_BRANCH}")
    return result


@app.command()
def audit(
    store_url: Annotated[str, typer.Argument(help="Public Icechunk store URL")],
    cutoff: Annotated[
        str | None,
        typer.Option(
            help="ISO timestamp with UTC offset; defaults to now minus one hour"
        ),
    ] = None,
    snapshot: Annotated[
        str | None,
        typer.Option(help="Snapshot ID; defaults to the tip of main"),
    ] = None,
    branch: Annotated[
        str | None,
        typer.Option(help="Branch tip to audit; mutually exclusive with --snapshot"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Report directory"),
    ] = None,
) -> None:
    """Probe a public store at an explicit read-only snapshot."""
    if snapshot is not None and branch is not None:
        raise typer.BadParameter("--snapshot and --branch are mutually exclusive")
    storage = _icechunk_storage(store_url)
    if storage is None:
        raise typer.BadParameter(f"Not an Icechunk store URL: {store_url}")
    config = (
        icechunk.Repository.fetch_config(storage) or icechunk.RepositoryConfig.default()
    )
    config.caching = icechunk.CachingConfig(num_chunk_refs=8_000_000)
    repo = icechunk.Repository.open(storage, config=config)
    result = _run_audit(
        repo,
        store_url,
        _parse_cutoff(cutoff),
        snapshot,
        branch,
        output or _default_output_dir(),
    )
    if result.present_keys:
        raise typer.Exit(1)


@app.command(
    help="Audit or delete refs on holdback-purge. Writing requires DYNAMICAL_ENV=prod "
    "and local Kubernetes access to the primary store's R2 secret."
)
def delete(
    dataset_id: Annotated[str, typer.Argument(help="Registered dataset ID")],
    cutoff: Annotated[
        str,
        typer.Option(help="Recorded ISO timestamp with UTC offset"),
    ],
    force: Annotated[
        bool,
        typer.Option(help="Delete and commit; the default is a dry run"),
    ] = False,
) -> None:
    """Audit or delete refs on the holdback-purge branch."""
    dataset = _operational_dataset(dataset_id)
    repo = dataset.store_factory.icechunk_primary_and_replica_repos()[0]
    _run_delete(
        repo,
        dataset.store_factory.primary_url(),
        _parse_cutoff(cutoff),
        _default_output_dir(),
        force=force,
    )


@app.command(
    help="Audit and atomically publish holdback-purge. Writing requires "
    "DYNAMICAL_ENV=prod and local Kubernetes access to its R2 secret."
)
def publish(
    dataset_id: Annotated[str, typer.Argument(help="Registered dataset ID")],
    from_snapshot: Annotated[
        str,
        typer.Option("--from-snapshot", help="Expected current main snapshot"),
    ],
    cutoff: Annotated[
        str,
        typer.Option(help="Recorded ISO timestamp with UTC offset"),
    ],
    force: Annotated[
        bool,
        typer.Option(help="Reset main and delete the purge branch after the audit"),
    ] = False,
) -> None:
    """Audit holdback-purge and publish it only with --force."""
    dataset = _operational_dataset(dataset_id)
    repo = dataset.store_factory.icechunk_primary_and_replica_repos()[0]
    _run_publish(
        repo,
        dataset.store_factory.primary_url(),
        from_snapshot,
        _parse_cutoff(cutoff),
        _default_output_dir(),
        force=force,
    )


if __name__ == "__main__":
    app()
