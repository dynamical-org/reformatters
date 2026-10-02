from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import batched, product
from math import ceil
from pathlib import Path
from typing import cast

import icechunk
import numpy as np
import pandas as pd
import zarr
from icechunk.store import IcechunkStore
from numpy.typing import NDArray
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common.logging import get_logger
from reformatters.common.virtual_region_job import _exists_many

from .holdback import PUBLICATION_HOLDBACK

_BATCH_SIZE = 20_000
log = get_logger(__name__)


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
        if not {"init_time", "lead_time"} <= set(dims) or not (
            {"y", "x"} <= set(dims) or {"latitude", "longitude"} <= set(dims)
        ):
            continue
        found_data_array = True
        if metadata.shards is not None:
            raise NotImplementedError(f"sharded arrays are unsupported: {array.path}")
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
    assert found_data_array, "no forecast data arrays found"


def probe_chunk_batches(
    store: IcechunkStore, chunks: Iterator[ForbiddenChunk]
) -> Iterator[tuple[Sequence[ForbiddenChunk], dict[str, bool]]]:
    for chunk_batch in batched(chunks, _BATCH_SIZE, strict=False):
        yield chunk_batch, _exists_many(store, [chunk.key for chunk in chunk_batch])


def summarize_probes(
    batches: Iterator[tuple[Sequence[ForbiddenChunk], dict[str, bool]]],
) -> tuple[int, int, dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts]]:
    total_keys = 0
    present_keys = 0
    counts: dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts] = {}
    for chunk_batch, presence in batches:
        present_batch = [chunk.key for chunk in chunk_batch if presence[chunk.key]]
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


def probe(
    store: IcechunkStore,
    group: zarr.Group,
    cutoff: pd.Timestamp,
) -> tuple[int, int, dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts]]:
    return summarize_probes(
        probe_chunk_batches(store, forbidden_chunk_keys(group, cutoff))
    )


def _utc_iso(value: pd.Timestamp | None) -> str:
    if value is None:
        return "none"
    assert value.tz is None
    return value.tz_localize("UTC").isoformat()


def write_report(
    *,
    store_label: str,
    snapshot: icechunk.SnapshotInfo,
    cutoff: pd.Timestamp,
    total_keys: int,
    present_keys: int,
    counts: dict[tuple[str, pd.Timestamp, pd.Timedelta], StepCounts],
    output_dir: Path,
    report_title: str = "WeatherNext publication holdback audit",
) -> HoldbackAudit:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = HoldbackAudit(
        total_keys=total_keys,
        present_keys=present_keys,
        counts=counts,
        report_path=output_dir / "holdback_audit.md",
    )
    header = [
        f"# {report_title}",
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


def run_audit(
    repo: icechunk.Repository,
    store_label: str,
    cutoff: pd.Timestamp,
    snapshot_id: str | None,
    branch: str | None,
    output_dir: Path,
    report_title: str = "WeatherNext publication holdback audit",
) -> HoldbackAudit:
    assert snapshot_id is None or branch is None
    if snapshot_id is None:
        snapshot_id = repo.lookup_branch(branch or "main")
    snapshot = repo.lookup_snapshot(snapshot_id)
    log.info(f"Snapshot: {snapshot.id} ({snapshot.written_at.isoformat()})")
    log.info(f"Cutoff: {_utc_iso(cutoff)}")
    store = repo.readonly_session(snapshot_id=snapshot.id).store
    group = zarr.open_group(store, mode="r")
    total_keys, present_keys, counts = probe(store, group, cutoff)
    return write_report(
        store_label=store_label,
        snapshot=snapshot,
        cutoff=cutoff,
        total_keys=total_keys,
        present_keys=present_keys,
        counts=counts,
        output_dir=output_dir,
        report_title=report_title,
    )
