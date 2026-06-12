# Virtual Datasets

A virtual dataset is an Icechunk store whose chunks are references — `(location, offset, length)` pointers into source files (e.g. GRIB messages) on the provider's object store — rather than copied bytes. A per-variable zarr serializer (e.g. `GribberishCodec`) decodes the referenced bytes at read time. Virtual datasets are written by `VirtualRegionJob` subclasses (`src/reformatters/common/virtual_region_job.py`).

How virtual jobs plug into worker parallelism and coordination is covered in [parallel_processing.md](parallel_processing.md); this doc covers what is specific to writing and reading virtual datasets.

## The write loop

`VirtualRegionJob.process_virtual` runs the same loop for backfills and operational updates; only the branch differs (a temp branch for backfill, `main` for operational):

1. `generate_source_file_coords` lists candidate source files for the region.
2. `filter_already_present` drops files whose refs are already in the manifest.
3. `process_virtual_refs(remaining)` — the generator each dataset implements — discovers available source files and yields batches of `(source file coord, its VirtualRefs)` pairs.
4. Each yield is committed atomically: open fresh writable sessions (a committed icechunk session is read-only), grow the append dim if needed (`sync_dims_to`), `set_virtual_refs`, commit.

**The yield is the commit unit.** The generator owns the batching policy: operational updates yield everything that arrived since the last poll tick (typically ~one file) for per-file visibility; backfills yield everything available. A yield must contain *whole* source files — never split a file across yields — and must never be empty (an empty icechunk commit raises). The loop asserts each file's refs cover the representative cell `filter_already_present` probes, so a file the filter could never see as ingested fails loudly instead of being re-ingested forever.

`processing_mode` on the job selects the generator's stopping rule: `"backfill"` sweeps what exists once and exits; `"update"` polls until everything expected is ingested, with the pod's active deadline bounding how long it waits on a file that never publishes. `operational_update_jobs` implementations must construct jobs with `processing_mode="update"` (asserted by the driver).

## Reader safety: one source file → one atomic commit

All refs from a single source file, plus any append-dim expansion their positions require, land in one icechunk commit. A reader on `main` sees either none of a file's data or all of it. Atomicity is per *file*, not per init — readers see each lead time's data as its file is ingested, within seconds.

This invariant is also what lets the filter probe a single representative cell per file: if one cell a file covers is present, the whole file is present.

## Operational updates: single writer

`operational_update_jobs()` returns **one** job spanning the active window (the recent, still-incomplete span of the append dim). One pod commits whole files straight to `main` as they arrive — no temp branch, no coordination files (`DynamicalDataset.update` routes to `_run_virtual_operational_update`).

Single-writer is what makes lazy append-dim expansion safe: `main` grows exactly to the data ingested (never NaN-padded toward "now"), and a resize can never race another writer. If an update falls far behind, run a backfill to catch up rather than scaling operational updates to multiple workers.

Crash recovery is automatic: committed refs are durable and the filter skips them on the next cron fire; uncommitted refs and expansions vanish and are re-discovered.

## Backfill: parallel on a pre-sized temp branch

Virtual backfills use the standard parallel temp-branch flow (see [parallel_processing.md](parallel_processing.md#icechunk-stores)). Worker 0 pre-sizes the full template on the branch, so every worker's `sync_dims_to` is a no-op and parallel workers write disjoint refs with no resize conflicts. Jobs partition by chunks along the append dim (virtual arrays have no shards).

Two operational rules when backfilling a live virtual dataset:

- **Suspend the dataset's `-update` CronJob for the duration of the backfill.** Finalize resets `main` to the temp branch only if `main` hasn't moved since setup; an operational fire committing to `main` mid-backfill would make finalize skip the reset (with a warning), discarding the backfill's work.
- **Choose `append_dim_end` as the last *fully published* position, not "now".** Finalize resets `main` to the pre-sized branch, so positions past the published data would appear as NaN-filled slots to readers.

## Filtering: the manifest is the source of truth

What is already ingested is derived from ref existence in the icechunk manifest (`store.exists(key)`), never from a progress coordinate or a chunk value read:

- A manifest probe has no false-positives (skipping undone work would be a permanent miss); a false-negative just re-emits an identical ref, which is idempotent.
- Never read or decode a virtual chunk to check presence (e.g. `.isnull()`) — that triggers a source-file download + decode per chunk.
- `representative_var(coord)` must return a variable the candidate file actually contains; override it for one-variable-per-file packings.

## Chunk keys

`chunk_key` maps a ref's coordinate labels to its zarr chunk index, shared by the filter and the emitter so they cannot disagree. Geometry (dim order, chunk sizes) is read from the checked-in template, and the filter's string keys are built with zarr's own `chunk_key_encoding` — no hand-rolled formats that could drift. A label not yet present in the coords means "not yet ingested."

## Replicas

The loop opens sessions on all stores and commits replicas-then-primary. "Committed" is defined by the primary (the filter probes the primary's manifest), so a crash between replica and primary commits is replayed idempotently on the next fire; replicas may briefly be a commit ahead, never behind.

## Storage and reading

A virtual dataset requires every storage config to use the `ICECHUNK` format and an `icechunk_virtual_config` registering the source buckets as virtual chunk containers (`DynamicalDataset` validates this). Container definitions are stored in the repo's persistent config; readers supply an (anonymous) credentials map via `authorize_virtual_chunk_access` when opening the store.
