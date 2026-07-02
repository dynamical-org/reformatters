import asyncio
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from typing import Any, ClassVar, Final, Generic, Literal, NamedTuple, cast

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from icechunk import VirtualChunkSpec
from icechunk.store import IcechunkStore
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common import storage
from reformatters.common.config_models import DataVar
from reformatters.common.iterating import node_group_name
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    DATA_VAR,
    SOURCE_FILE_COORD,
    CoordinateValue,
    RegionJob,
    SourceFileResult,
)
from reformatters.common.types import Dim, Timedelta

log = get_logger(__name__)


class VirtualRef(NamedTuple):
    """A single virtual chunk reference: one GRIB message -> one zarr chunk.
    `out_loc` is the output cell the message fills (e.g. {init_time, lead_time}).
    """

    data_var: DataVar[Any]
    out_loc: Mapping[Dim, CoordinateValue]
    location: str
    offset: int
    length: int


class VirtualRegionJob(
    RegionJob[DATA_VAR, SOURCE_FILE_COORD], Generic[DATA_VAR, SOURCE_FILE_COORD]
):
    """Base class for processing a region of virtual Icechunk datasets that point to external files.

    See docs/virtual_datasets.md for the write loop, filtering, and reader-safety guarantees.
    """

    # ----- Class attributes -----

    # Locked to None: an int would split one file's vars across independently
    # committing jobs, breaking the per-file commit atomicity readers rely on.
    max_vars_per_job: ClassVar[Final[int | None]] = None

    # Contiguous blocks keep each worker's commits within 1-2 manifest windows per array
    # (scattered regions rewrite most windows every flush), see docs/parallel_processing.md.
    worker_assignment: ClassVar[Literal["spread", "contiguous"]] = "contiguous"

    # Updates wait for source files as the provider publishes them, backfills check once
    processing_mode: Literal["backfill", "update"] = "backfill"

    # When polling, pace each discovery sweep to at most one per tick.
    tick_interval: ClassVar[Timedelta] = pd.Timedelta("1s")
    # Concurrent file downloads while building refs; small .idx files, so IO-bound.
    download_concurrency: ClassVar[int] = 64

    # ----- Overridable methods -----
    # A dataset implements file_refs and generate_source_file_coords (from
    # RegionJob); the rest have working defaults.

    def file_refs(self, coord: SOURCE_FILE_COORD, file_size: int) -> list[VirtualRef]:
        """Build every virtual ref a single source file contributes (or [] to skip it).

        Resolve each message's byte range however the source allows — parse a sidecar
        index (`coord.get_index_url()`), scan the data file, or (one message per file)
        point at the whole file — and return a VirtualRef per (out cell, variable) in
        coordinate-label space; the chunk index is resolved centrally later (see
        _emit_refs). Return [] to drop an unreadable or stale file. `file_size` is
        whatever discover_available reported (a listed size, a Content-Length), e.g.
        to supply the missing final end byte of an index that omits it.
        """
        raise NotImplementedError(
            "Return the VirtualRefs for one source file, or [] to skip it."
        )

    def discover_available(
        self, pending: list[SOURCE_FILE_COORD]
    ) -> list[tuple[SOURCE_FILE_COORD, int]]:
        """Of the not-yet-present files, the subset fetchable now, each with its data-file size.

        Return the same coord objects from `pending` (the loop drops them by
        identity). For an obstore-listable backend, delegate to
        `virtual_source_listing.discover_available_by_obstore_listing`; for a source
        obstore can't list (an HTML directory index, a frontier to probe,
        assume-all) implement it directly.
        """
        raise NotImplementedError(
            "Return the (coord, file_size) pairs ready to fetch now."
        )

    def representative_var(self, coord: SOURCE_FILE_COORD) -> DATA_VAR:
        """The variable whose chunk presence means `coord`'s file is fully ingested.

        Probed by filter_already_present and asserted by the write loop, so it must
        be a variable the file's refs actually cover. Default: the first instant var
        (most likely to have data at every step) among the vars the file carries —
        `coord.data_vars` if it declares which subset it packs, else all of
        self.data_vars — falling back to the first such var. Override only for a
        packing neither rule captures (e.g. one variable per file keyed off the
        coord rather than its data_vars).
        """
        file_vars = getattr(coord, "data_vars", None) or self.data_vars
        return next(
            (var for var in file_vars if var.attrs.step_type == "instant"),
            file_vars[0],
        )

    def filter_already_present(
        self,
        candidates: Sequence[SOURCE_FILE_COORD],
        store: IcechunkStore,
    ) -> list[SOURCE_FILE_COORD]:
        """Drop candidates whose representative chunk is already in the manifest.

        Probes ref existence (store.exists) - never reads or decodes a chunk, which
        would trigger a download + decode.
        """
        group = zarr.open_group(store, mode="r")
        rep_vars = [self.representative_var(coord) for coord in candidates]
        indices = self._resolve_chunk_keys(
            [
                (coord.out_loc(), var)
                for coord, var in zip(candidates, rep_vars, strict=True)
            ]
        )
        keyed: list[tuple[SOURCE_FILE_COORD, str | None]] = []
        for coord, var, index in zip(candidates, rep_vars, indices, strict=True):
            if index is None:
                keyed.append((coord, None))
                continue
            array = group[var.path]
            assert isinstance(array, zarr.Array)
            metadata = array.metadata
            assert isinstance(metadata, ArrayV3Metadata)
            key = f"{array.path}/{metadata.chunk_key_encoding.encode_chunk_key(index)}"
            keyed.append((coord, key))

        present = _exists_many(store, [key for _, key in keyed if key is not None])
        return [coord for coord, key in keyed if key is None or not present[key]]

    # ----- Common write-loop machinery: subclasses do not implement these -----

    def process_virtual_refs(
        self,
        remaining: Sequence[SOURCE_FILE_COORD],
    ) -> Iterator[Sequence[tuple[SOURCE_FILE_COORD, Sequence[VirtualRef]]]]:
        """Drive discover_available + file_refs, yielding one commit's worth per tick.

        One yield per tick contains every file that became available since the last:
        a backfill sweeps once and exits, an update polls until everything is
        ingested. Each yield is whole source files as (coord, refs) pairs — never
        split a file, never yield empty. Source-agnostic: it only asks
        discover_available which coords are ready. Override only for a different
        batching policy. See "The write loop" in docs/virtual_datasets.md.
        """
        pending = list(remaining)
        with ThreadPoolExecutor(self.download_concurrency) as pool:
            while pending:
                tick_start = time.monotonic()
                available = self.discover_available(pending)
                discover_s = time.monotonic() - tick_start
                if available:
                    coords, sizes = zip(*available, strict=True)
                    build_start = time.monotonic()
                    refs_per_file = pool.map(self._file_refs_or_skip, coords, sizes)
                    # Drop files that yielded no refs (skipped as unreadable).
                    batch = [
                        (coord, refs)
                        for coord, refs in zip(coords, refs_per_file, strict=True)
                        if refs
                    ]
                    build_s = time.monotonic() - build_start
                    ready = {id(coord) for coord in coords}
                    pending = [coord for coord in pending if id(coord) not in ready]
                    skipped = len(coords) - len(batch)
                    log.info(
                        f"Ingesting {len(batch)} files"
                        f"{f' ({skipped} skipped)' if skipped else ''}, "
                        f"{len(pending)} still pending "
                        f"(discover {discover_s:.1f}s, build {build_s:.1f}s)"
                    )
                    if batch:
                        yield batch
                if self.processing_mode == "backfill":
                    if pending:
                        log.info(
                            f"{len(pending)} source files not present, skipping "
                            f"(first: {pending[0].get_url()})"
                        )
                    return
                if pending:
                    elapsed = time.monotonic() - tick_start
                    time.sleep(max(0.0, self.tick_interval.total_seconds() - elapsed))

    def _file_refs_or_skip(
        self, coord: SOURCE_FILE_COORD, file_size: int
    ) -> list[VirtualRef]:
        # Skip a file we can't build refs for rather than sink the job; AssertionError
        # is our own invariant, so let it propagate.
        try:
            return self.file_refs(coord, file_size)
        except AssertionError:
            raise
        except Exception:
            log.exception(f"Skipping {coord.get_url()}: could not build virtual refs")
            return []

    @classmethod
    def process_worker_jobs(
        cls,
        worker_jobs: Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]],
        store_factory: storage.StoreFactory,
        branch_name: str,
        worker_index: int,  # noqa: ARG003 - per-batch commit messages don't carry it
    ) -> dict[str, list[SourceFileResult]]:
        """Drive the whole worker's virtual write loop on ``branch_name``.

        Gathers the not-already-present source files across all the worker's jobs
        against a single readonly view, then runs one write loop over their union:
        a backfill worker's generator yields a single batch (one commit for the
        whole worker), an update job's generator polls and commits per tick.
        Always returns an empty dict: virtual refs live in the icechunk manifest,
        so there is nothing to thread back to finalize.
        """
        assert worker_jobs, "process_worker_jobs requires at least one job"
        assert all(isinstance(job, VirtualRegionJob) for job in worker_jobs)
        jobs = cast(
            "Sequence[VirtualRegionJob[DATA_VAR, SOURCE_FILE_COORD]]", worker_jobs
        )
        del worker_jobs

        primary_repo, replica_repos = store_factory.icechunk_primary_and_replica_repos()
        readonly_store = primary_repo.readonly_session(branch_name).store
        remaining = [
            coord
            for job in jobs
            for coord in job.filter_already_present(
                job.source_file_coords(), readonly_store
            )
        ]
        # An all-already-present worker writes nothing; an empty icechunk commit
        # would raise, so skip the write loop entirely.
        if remaining:
            # A worker's jobs share template_ds/processing_mode/tick_interval, so any
            # one drives the write loop over the union of their coords. Poison the
            # driver's region: the loop spans every job's region, so reading a single
            # job's region would be a bug (see _NoRegion).
            driver = jobs[0].model_copy(update={"region": _NO_REGION})
            driver.process_virtual(
                primary_repo, list(replica_repos), branch_name, remaining
            )
        return {}

    def process_virtual(
        self,
        primary_repo: icechunk.Repository,
        replica_repos: Sequence[icechunk.Repository],
        branch: str,
        remaining: Sequence[SOURCE_FILE_COORD],
    ) -> None:
        """Run the virtual write loop over `remaining`, committing each yielded batch atomically.

        `remaining` is the pre-filtered union of the worker's not-already-present
        source files (gathered by process_worker_jobs). The same loop serves
        backfill (temp branch, one batch -> one commit) and the single-writer
        operational update ("main", per-tick commits); see "The write loop" in
        docs/virtual_datasets.md.
        """
        # A fresh writable session is opened per batch because an icechunk
        # session becomes read-only after commit. sync_dims_to grows the store
        # lazily (a no-op on the pre-sized backfill branch).
        readonly_store = primary_repo.readonly_session(branch).store
        # The smallest group, so the sync_dims_to shortcut below can never skip a group
        # that lags root (e.g. after a partially-applied prior grow).
        current_size = min(
            self._append_dim_size(readonly_store, node_group_name(node))
            for node in self.template_ds.subtree
        )

        for file_refs_batch in self.process_virtual_refs(remaining):
            assert file_refs_batch, "process_virtual_refs yielded an empty batch"
            for coord, file_refs in file_refs_batch:
                self._assert_probe_chunk_covered(coord, file_refs)
            refs = [ref for _, file_refs in file_refs_batch for ref in file_refs]
            primary_session = primary_repo.writable_session(branch)
            replica_sessions = [repo.writable_session(branch) for repo in replica_repos]
            stores = [primary_session.store, *(s.store for s in replica_sessions)]

            needed_size = self._needed_append_dim_size(refs)
            if needed_size > current_size:
                self.sync_dims_to(stores, needed_size)
                current_size = needed_size

            emit_start = time.monotonic()
            self._emit_refs(stores, refs)
            emit_s = time.monotonic() - emit_start

            now = pd.Timestamp.now(tz="UTC")
            commit_start = time.monotonic()
            storage.commit_if_icechunk(
                f"Update at {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                primary_session.store,
                [s.store for s in replica_sessions],
            )
            log.info(
                f"Committed {len(refs)} refs "
                f"(emit {emit_s:.1f}s, commit {time.monotonic() - commit_start:.1f}s)"
            )

    def _assert_probe_chunk_covered(
        self, coord: SOURCE_FILE_COORD, file_refs: Sequence[VirtualRef]
    ) -> None:
        """A file's refs must cover the chunk filter_already_present probes
        (representative_var at the file's out_loc)."""
        assert file_refs, f"empty refs for source file {coord}"
        rep = self.representative_var(coord)
        probe = self.chunk_key(coord.out_loc(), rep)
        assert any(
            ref.data_var.path == rep.path and self.chunk_key(ref.out_loc, rep) == probe
            for ref in file_refs
        ), (
            f"refs for {coord} do not cover representative chunk "
            f"({rep.name}, {dict(coord.out_loc())}); the filter would re-ingest "
            "this file forever. Override representative_var to pick a variable "
            "the file actually contains."
        )

    def _emit_refs(
        self, stores: Sequence[IcechunkStore], refs: Sequence[VirtualRef]
    ) -> None:
        indices = self._resolve_chunk_keys(
            [(ref.out_loc, ref.data_var) for ref in refs]
        )
        specs_by_var: dict[str, list[VirtualChunkSpec]] = {}
        for ref, index in zip(refs, indices, strict=True):
            assert index is not None, (
                f"ref {ref.data_var.name} {dict(ref.out_loc)} did not resolve to a "
                "chunk index after expansion"
            )
            specs_by_var.setdefault(ref.data_var.path, []).append(
                VirtualChunkSpec(
                    index=list(index),
                    location=ref.location,
                    offset=ref.offset,
                    length=ref.length,
                )
            )
        for store in stores:
            for array_path, specs in specs_by_var.items():
                # array_path is group-qualified (e.g. "pressure_level/temperature").
                failed = store.set_virtual_refs(
                    array_path, specs, validate_containers=True
                )
                assert failed is None, (
                    f"{len(failed)} virtual ref(s) for {array_path} did not match a "
                    f"registered container, e.g. {failed[:3]}"
                )

    def chunk_key(
        self, out_loc: Mapping[Dim, CoordinateValue], var: DataVar[Any]
    ) -> tuple[int, ...] | None:
        """Map a source message's coordinate labels to its zarr chunk index.

        Returns None if a label is not in the template's coords (the filter treats
        that as "remaining"). A batch of one through _resolve_chunk_keys, which
        holds the actual chunk-index math -- see there.
        """
        return self._resolve_chunk_keys([(out_loc, var)])[0]

    def _resolve_chunk_keys(
        self, items: Sequence[tuple[Mapping[Dim, CoordinateValue], DataVar[Any]]]
    ) -> list[tuple[int, ...] | None]:
        """Map each (out_loc, var) pair to its zarr chunk index, or None if a label
        isn't in the template's coords yet (the filter treats that as "remaining").

        Groups items by var.path and, per group, resolves every labeled dim's
        position with one vectorized pandas.Index.get_indexer call over the whole
        group, rather than one call per item. chunk_key is a batch-of-one wrapper
        around this.
        """
        order = sorted(range(len(items)), key=lambda i: items[i][1].path)
        results: list[tuple[int, ...] | None] = [None] * len(items)

        for var_path, idx_group in groupby(order, key=lambda i: items[i][1].path):
            idxs = list(idx_group)
            template_var = self.template_ds[var_path]
            dims = tuple(str(d) for d in template_var.dims)
            chunks = tuple(template_var.encoding["chunks"])
            assert len(chunks) == len(dims)

            group_labels = [
                {str(dim): label for dim, label in items[i][0].items()} for i in idxs
            ]
            labeled_dims = set(group_labels[0])
            assert all(set(labels) == labeled_dims for labels in group_labels), (
                f"every ref for {var_path} must label the same set of dims"
            )

            n = len(idxs)
            chunk_indices = np.zeros((n, len(dims)), dtype=np.int64)
            present = np.ones(n, dtype=bool)
            for dim_i, (dim, chunk_size) in enumerate(zip(dims, chunks, strict=True)):
                if dim in labeled_dims:
                    labels = pd.Index([lbl[dim] for lbl in group_labels])
                    positions = template_var.get_index(dim).get_indexer(labels)
                    present &= positions >= 0
                    chunk_idx, remainder = np.divmod(
                        np.where(positions >= 0, positions, 0), chunk_size
                    )
                    assert np.all(remainder[positions >= 0] == 0), (
                        f"a ref's {dim} label does not fall on a chunk boundary "
                        f"(chunk size {chunk_size}); a virtual chunk must be exactly "
                        "one GRIB message."
                    )
                else:
                    # A dim absent from out_loc must be single-chunk, else all refs
                    # collapse to chunk 0 along it.
                    size = int(template_var.sizes[dim])
                    assert chunk_size >= size, (
                        f"dim {dim} is absent from out_loc but spans multiple chunks "
                        f"(size {size}, chunk {chunk_size}); out_loc must locate every "
                        "multi-chunk dim."
                    )
                    chunk_idx = np.zeros(n, dtype=np.int64)
                chunk_indices[:, dim_i] = chunk_idx

            for local_i, item_i in enumerate(idxs):
                results[item_i] = (
                    tuple(int(v) for v in chunk_indices[local_i])
                    if present[local_i]
                    else None  # label not in coords -> not yet a position in this dataset
                )
        return results

    def sync_dims_to(
        self, stores: Sequence[IcechunkStore], needed_append_dim_size: int
    ) -> None:
        """Grow the append dim (and its dependent coords) to `needed_append_dim_size`
        in every group (DataTree.to_zarr has no append_dim, so each node is appended
        on its own)."""
        # compute=False keeps data vars dask-lazy so the decode-only serializer is
        # never invoked. Each (store, group)'s missing slice is read from its own
        # committed size.
        for store in stores:
            for node in self.template_ds.subtree:
                group = node_group_name(node)
                current_size = self._append_dim_size(store, group)
                if needed_append_dim_size <= current_size:
                    continue
                slice_ds = node.to_dataset().isel(
                    {self.append_dim: slice(current_size, needed_append_dim_size)}
                )
                slice_ds.to_zarr(
                    store,
                    group=group,
                    append_dim=self.append_dim,
                    compute=False,
                    consolidated=False,
                )

    def _needed_append_dim_size(self, refs: Sequence[VirtualRef]) -> int:
        positions = (
            self.template_ds.coords[self.append_dim]
            .to_index()
            .get_indexer(pd.Index([ref.out_loc[self.append_dim] for ref in refs]))
        )
        assert (positions >= 0).all(), (
            "a ref's append-dim label is not present in the template"
        )
        return int(positions.max()) + 1

    def _append_dim_size(self, store: IcechunkStore, group: str | None = None) -> int:
        node = zarr.open_group(store, mode="r")
        if group is not None:
            node = node[group]
            assert isinstance(node, zarr.Group)
        array = node[self.append_dim]
        assert isinstance(array, zarr.Array)
        return int(array.shape[0])

    def source_file_coords(self) -> Sequence[SOURCE_FILE_COORD]:
        """Every source file coord this job's region covers — the write loop's inputs,
        before filtering. Reused by operational validation to probe manifest completeness."""
        return self.generate_source_file_coords(
            self._processing_region_ds(), self.data_vars
        )

    def _processing_region_ds(self) -> xr.Dataset:
        processing_region = self.get_processing_region()
        # Virtual refs point at raw source bytes, so no surrounding buffer is needed.
        assert processing_region == self.region, (
            "VirtualRegionJob.get_processing_region must equal self.region"
        )
        # generate_source_file_coords reads only coordinate values, so pass the region's
        # coords from every group — never the data vars, whose names can collide across
        # vertical groups.
        coords = {
            name: coord
            for node in self.template_ds.subtree
            for name, coord in node.to_dataset().coords.items()
        }
        return xr.Dataset(coords=coords).isel({self.append_dim: processing_region})


class _NoRegion:
    """Poison `region` for the batched write-loop driver (see process_worker_jobs).

    process_worker_jobs runs one write loop over the union of all its jobs'
    coords, so the driver's own region is meaningless. process_virtual and every
    method it calls must act on the coords passed to them, never self.region;
    reading it would silently narrow the batch to a single job. Any access raises
    instead, so a future edit that reintroduces region-dependence fails loudly.
    """

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        raise AssertionError(
            "the batched virtual write loop read self.region; process_virtual and "
            "the methods it calls must use the coords passed to them, not self.region"
        )


_NO_REGION: Final = _NoRegion()


def _exists_many(store: IcechunkStore, keys: Sequence[str]) -> dict[str, bool]:
    """Probe many chunk keys concurrently (store.exists is async)."""
    if not keys:
        return {}

    async def _check() -> list[bool]:
        return list(await asyncio.gather(*(store.exists(key) for key in keys)))

    return dict(zip(keys, asyncio.run(_check()), strict=True))
