import asyncio
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar, Final, Generic, NamedTuple

import icechunk
import pandas as pd
import xarray as xr
import zarr
from icechunk import VirtualChunkSpec
from icechunk.store import IcechunkStore
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common import storage
from reformatters.common.config_models import DataVar
from reformatters.common.region_job import (
    DATA_VAR,
    SOURCE_FILE_COORD,
    CoordinateValue,
    RegionJob,
    SourceFileResult,
)
from reformatters.common.types import Dim


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

    # Locked to None: an integer max_vars_per_job would split one source file's
    # variables across separate jobs that commit independently, breaking the
    # per-file commit atomicity virtual readers rely on.
    max_vars_per_job: ClassVar[Final[int | None]] = None

    def process_virtual_refs(
        self,
        remaining: Sequence[SOURCE_FILE_COORD],
    ) -> Iterator[Sequence[VirtualRef]]:
        """Discover available source files among `remaining` and yield their virtual references.

        Each yield is one commit's worth of *whole* source files — never split a
        file across yields, never yield an empty batch. The generator owns the
        batching policy. See "The write loop" in docs/virtual_datasets.md.
        """
        raise NotImplementedError(
            "Yield batches of VirtualRefs, each one commit's worth of whole source files."
        )

    def representative_var(self, coord: SOURCE_FILE_COORD) -> DATA_VAR:  # noqa: ARG002
        """The variable whose chunk presence means `coord`'s file is fully ingested.

        Used by filter_already_present to probe a single representative cell per
        file. Default: the first instant var (most likely to have data at every
        step), else the first data var; valid when every file contains every
        variable. Override for one-variable-per-file packings (e.g. GEFS v12
        reforecast) to return the candidate's own variable.
        """
        return next(
            (var for var in self.data_vars if var.attrs.step_type == "instant"),
            self.data_vars[0],
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
        keyed: list[tuple[SOURCE_FILE_COORD, str | None]] = []
        for coord in candidates:
            var = self.representative_var(coord)
            index = self.chunk_key(coord.out_loc(), var)
            if index is None:
                keyed.append((coord, None))
                continue
            array = group[var.name]
            assert isinstance(array, zarr.Array)
            metadata = array.metadata
            assert isinstance(metadata, ArrayV3Metadata)
            key = f"{array.path}/{metadata.chunk_key_encoding.encode_chunk_key(index)}"
            keyed.append((coord, key))

        present = _exists_many(store, [key for _, key in keyed if key is not None])
        return [coord for coord, key in keyed if key is None or not present[key]]

    # ----- Most subclasses will not need to override the attributes and methods below -----

    @classmethod
    def process_worker_jobs(
        cls,
        worker_jobs: Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]],
        store_factory: storage.StoreFactory,
        branch_name: str,
        worker_index: int,  # noqa: ARG003 - virtual jobs commit per batch with their own messages
    ) -> dict[str, list[SourceFileResult]]:
        """Drive each job's per-batch virtual write loop on ``branch_name``.

        Always returns an empty dict of results: virtual refs live in the icechunk
        manifest, so there is nothing to thread back to finalize.
        """
        assert worker_jobs, "process_worker_jobs requires at least one job"
        primary_repo, replica_repos = store_factory.icechunk_primary_and_replica_repos()
        for job in worker_jobs:
            assert isinstance(job, VirtualRegionJob)
            job.process_virtual(primary_repo, replica_repos, branch_name)
        return {}

    def process_virtual(
        self,
        primary_repo: icechunk.Repository,
        replica_repos: Sequence[icechunk.Repository],
        branch: str,
    ) -> None:
        """Run the virtual write loop, committing each yielded batch atomically.

        The same loop serves backfill (temp branch) and the single-writer
        operational update ("main"); see "The write loop" in docs/virtual_datasets.md.
        """
        # A fresh writable session is opened per batch because an icechunk
        # session becomes read-only after commit. sync_dims_to grows the store
        # lazily (a no-op on the pre-sized backfill branch).
        readonly_store = primary_repo.readonly_session(branch).store
        candidates = self.generate_source_file_coords(
            self._processing_region_ds(), self.data_vars
        )
        remaining = self.filter_already_present(candidates, readonly_store)
        current_size = self._append_dim_size(readonly_store)

        for refs in self.process_virtual_refs(remaining):
            # The generator yields whole files, so a batch always has refs; emitting
            # them always dirties the session, so the commit below is never empty
            # (an empty icechunk commit raises rather than no-ops).
            assert refs, "process_virtual_refs yielded an empty batch"
            primary_session = primary_repo.writable_session(branch)
            replica_sessions = [repo.writable_session(branch) for repo in replica_repos]
            stores = [primary_session.store, *(s.store for s in replica_sessions)]

            needed_size = self._needed_append_dim_size(refs)
            if needed_size > current_size:
                self.sync_dims_to(stores, needed_size)
                current_size = needed_size

            self._emit_refs(stores, refs)

            now = pd.Timestamp.now(tz="UTC")
            storage.commit_if_icechunk(
                f"Update at {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                primary_session.store,
                [s.store for s in replica_sessions],
            )

    def _emit_refs(
        self, stores: Sequence[IcechunkStore], refs: Sequence[VirtualRef]
    ) -> None:
        specs_by_var: dict[str, list[VirtualChunkSpec]] = {}
        for ref in refs:
            index = self.chunk_key(ref.out_loc, ref.data_var)
            assert index is not None, (
                f"ref {ref.data_var.name} {dict(ref.out_loc)} did not resolve to a "
                "chunk index after expansion"
            )
            specs_by_var.setdefault(ref.data_var.name, []).append(
                VirtualChunkSpec(
                    index=list(index),
                    location=ref.location,
                    offset=ref.offset,
                    length=ref.length,
                )
            )
        for store in stores:
            for var_name, specs in specs_by_var.items():
                failed = store.set_virtual_refs(
                    var_name, specs, validate_containers=True
                )
                assert failed is None, (
                    f"{len(failed)} virtual ref(s) for {var_name} did not match a "
                    f"registered container, e.g. {failed[:3]}"
                )

    def chunk_key(
        self, out_loc: Mapping[Dim, CoordinateValue], var: DataVar[Any]
    ) -> tuple[int, ...] | None:
        """Map a source message's coordinate labels to its zarr chunk index.

        Returns None if a label is not in the template's coords, which the filter
        treats as "remaining". Asserts each label lands on an exact chunk boundary -
        a virtual chunk is exactly one GRIB message, so any non-aligned position is
        a bug.
        """
        # Geometry (dim order, chunk sizes, coord positions) is read from the
        # checked-in template (the authoritative metadata, git-reviewable), not the
        # in-code DataVar.encoding which could drift from it. So the index matches
        # what readers resolve and does not depend on how far the store has been
        # expanded. Shared by the filter and the emitter so they cannot disagree.
        template_var = self.template_ds[var.name]
        dims = tuple(str(d) for d in template_var.dims)
        chunks = template_var.encoding["chunks"]
        assert len(chunks) == len(dims)
        labels = {str(dim): label for dim, label in out_loc.items()}

        index: list[int] = []
        for dim, chunk_size in zip(dims, chunks, strict=True):
            if dim in labels:
                position = int(
                    self.template_ds.get_index(dim).get_indexer(
                        pd.Index([labels[dim]])
                    )[0]
                )
                if position < 0:
                    return None  # label not in coords -> not yet a position in this dataset
            else:
                # A dim absent from out_loc (e.g. lat/lon) must be a single chunk;
                # otherwise every ref would collapse to chunk 0 along it (later refs
                # overwriting earlier, the rest left as permanent fill-value holes).
                # A multi-chunk dim (e.g. ensemble_member packed one-per-file) must
                # appear in out_loc so it resolves to the right chunk.
                assert chunk_size >= template_var.sizes[dim], (
                    f"dim {dim} is absent from out_loc but spans multiple chunks "
                    f"(size {template_var.sizes[dim]}, chunk {chunk_size}); out_loc "
                    "must locate every multi-chunk dim."
                )
                position = 0
            chunk_index, remainder = divmod(position, chunk_size)
            assert remainder == 0, (
                f"{dim}={labels.get(dim)} maps to position {position}, which is not "
                f"on a chunk boundary (chunk size {chunk_size}); a virtual chunk must "
                "be exactly one GRIB message."
            )
            index.append(chunk_index)
        return tuple(index)

    def sync_dims_to(
        self, stores: Sequence[IcechunkStore], needed_append_dim_size: int
    ) -> None:
        """Grow the append dim (and its dependent coords) to `needed_append_dim_size`."""
        # Writes only the new positions by appending the corresponding slice of the
        # already-derived template; existing positions are untouched. Data vars stay
        # dask-lazy under compute=False, so the decode-only serializer is never
        # invoked. A no-op for any store already covering the size (so backfill's
        # pre-sized branch never resizes, and parallel workers never conflict). Each
        # store's missing slice is computed from its own committed size: replicas
        # commit before the primary, so a partial commit can leave a replica ahead,
        # and appending the primary's slice to it would duplicate positions.
        for store in stores:
            current_size = self._append_dim_size(store)
            if needed_append_dim_size <= current_size:
                continue
            slice_ds = self.template_ds.isel(
                {self.append_dim: slice(current_size, needed_append_dim_size)}
            )
            slice_ds.to_zarr(
                store, append_dim=self.append_dim, compute=False, consolidated=False
            )

    def _needed_append_dim_size(self, refs: Sequence[VirtualRef]) -> int:
        positions = self.template_ds.get_index(self.append_dim).get_indexer(
            pd.Index([ref.out_loc[self.append_dim] for ref in refs])
        )
        assert (positions >= 0).all(), (
            "a ref's append-dim label is not present in the template"
        )
        return int(positions.max()) + 1

    def _append_dim_size(self, store: IcechunkStore) -> int:
        array = zarr.open_group(store, mode="r")[self.append_dim]
        assert isinstance(array, zarr.Array)
        return int(array.shape[0])

    def _processing_region_ds(self) -> xr.Dataset:
        processing_region = self.get_processing_region()
        # Virtual refs point at raw source bytes, so no surrounding buffer is ever
        # needed; a buffered region would make adjacent backfill workers generate
        # overlapping candidates and emit refs into each other's regions.
        assert processing_region == self.region, (
            "VirtualRegionJob.get_processing_region must equal self.region"
        )
        ds: xr.Dataset = self.template_ds[[v.name for v in self.data_vars]]  # ty: ignore[invalid-assignment]
        return ds.isel({self.append_dim: processing_region})


def _exists_many(store: IcechunkStore, keys: Sequence[str]) -> dict[str, bool]:
    """Probe many chunk keys concurrently (store.exists is async)."""
    if not keys:
        return {}

    async def _check() -> list[bool]:
        return list(await asyncio.gather(*(store.exists(key) for key in keys)))

    return dict(zip(keys, asyncio.run(_check()), strict=True))
