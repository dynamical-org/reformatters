# Virtual Datasets

A virtual dataset is an Icechunk store whose chunks are references — `(location, offset, length)` pointers into source files (e.g. GRIB messages) on the provider's object store — rather than copied bytes. A per-variable zarr serializer (e.g. `GribberishCodec`) decodes the referenced bytes at read time. Virtual datasets are written by `VirtualRegionJob` subclasses (`src/reformatters/common/virtual_region_job.py`).

How virtual jobs plug into worker parallelism and coordination is covered in [parallel_processing.md](parallel_processing.md); this doc covers what is specific to writing and reading virtual datasets.

## Why virtual

A virtual dataset complements the materialized (rechunked) time-series dataset over the same source data:

- **Spatial / map-optimized chunking.** Chunks follow the native GRIB message shape — one time step, full spatial grid — so a virtual `-spatial` dataset serves map and spatial queries efficiently, while the materialized dataset stays tuned for time-series extraction.
- **Every curated source variable, cheaply.** A variable costs only a handful of byte-range refs, not a rechunked copy, so a virtual dataset can include every variable we have metadata for rather than just the materialized subset.
- **Very low latency updates.** Writing refs records byte offsets without moving data, so a new source file's data is visible within seconds of publication (target ≤ 5 s end to end). See [Operational updates](#operational-updates-single-writer).

## Scale

Virtual datasets are *metadata-heavy*, not storage-heavy. One GRIB message — one `(variable, …, lead/time)` field — becomes one virtual chunk, so ref counts track the full Cartesian product of the dataset's dimensions *including the variable axis*. A multi-year ensemble dataset reaches order 10⁹ refs and tens of millions of source index files. Two consequences shape the design:

- **Manifests must be split** (see [Storage](#storage-and-reading)) — a single per-array manifest at this ref count is too large to rewrite on every commit.
- **Backfills must run in parallel** — backfill is bounded by index-file *count*, not ref count, and tens of millions of index downloads is hours of work even at high concurrency. Operational updates touch a handful of files per fire and run single-writer.

## The write loop

`process_worker_jobs` gathers the not-already-present source files across all of a worker's region jobs (one readonly view, one filter pass) into `remaining`, then `VirtualRegionJob.process_virtual` runs the same write loop over that union for both backfills and operational updates; only the branch differs (a temp branch for backfill, `main` for operational):

1. `generate_source_file_coords` lists candidate source files for each of the worker's region jobs.
2. `filter_already_present` drops files whose refs are already in the manifest; the survivors across the worker's jobs are unioned into `remaining`.
3. `process_virtual_refs(remaining)` — a concrete `VirtualRegionJob` generator — each tick asks `discover_available` which files are ready, builds their refs with `file_refs`, and yields batches of `(source file coord, its VirtualRefs)` pairs.
4. Each yield is committed atomically: open fresh writable sessions (a committed icechunk session is read-only), grow the append dim if needed (`sync_dims_to`), `set_virtual_refs`, commit.

**The yield is the commit unit.** The loop's batching policy: a backfill sweeps once and yields everything available across the worker's jobs, so the whole worker is **one commit** — this keeps the commit count at `workers_total` rather than one-per-init, relieving shared-branch CAS contention (`jobs_per_pod` sets the batch = inits per commit, mirroring `MaterializedRegionJob`). An operational update yields everything that arrived since the last poll tick (typically ~one file) for per-file visibility. A yield contains *whole* source files — never split a file across yields — and is never empty (an empty icechunk commit raises). The loop asserts each file's refs cover the representative cell `filter_already_present` probes, so a file the filter could never see as ingested fails loudly instead of being re-ingested forever.

`processing_mode` on the job selects the generator's stopping rule: `"backfill"` sweeps what exists once and exits; `"update"` polls until everything expected is ingested, with the pod's active deadline bounding how long it waits on a file that never publishes. `operational_update_jobs` implementations must construct jobs with `processing_mode="update"` (asserted by the driver).

## What a `VirtualRegionJob` implementer writes

`VirtualRegionJob` is **source-agnostic**: the base owns the tick loop (`process_virtual_refs`, working in coord space), the atomic-commit machinery, lazy append-dim expansion (`sync_dims_to`), central chunk-index resolution (`chunk_key`/`_emit_refs`), and unreadable-file skipping (`_file_refs_or_skip`). It assumes nothing about S3, object listing, or even that a source file *has* an index. A dataset fills in the seam.

**The core three.** Most of a virtual dataset is three methods, run in this order to drive an update:

1. `generate_source_file_coords(processing_region_ds, data_var_group) -> [coord]` — list every source file this job covers.
2. `discover_available(pending) -> [(coord, file_size)]` — given the source files not yet ingested, return those available to fetch right now, each with its data-file size. Obstore-listable backends one-line this via `discover_available_by_obstore_listing` (below); other sources implement it directly.
3. `file_refs(coord, file_size) -> [VirtualRef]` — given one available file, return every pointer it contributes — a `VirtualRef` is `(source location, byte offset, length) -> one output chunk` — or `[]` to skip the file. Resolve byte ranges however the source allows: parse a sidecar index, scan the data file, or (one message per file) point at the whole file. Refs are in coordinate-label space; the chunk index is resolved centrally.

(Between 1 and 2 the base runs `filter_already_present` to drop files already in the manifest; steps 2–3 then repeat each tick until everything is ingested. See "The write loop" above.)

**Also required:** `operational_update_jobs(...)` — the update factory (per-subclass, like materialized datasets); encodes the re-sweep window and constructs the job with `processing_mode="update"`.

**On the `SourceFileCoord` subclass** (identifies one source file):

- `get_url() -> str` — the canonical data-file URL refs point at (must match the virtual chunk container prefix).
- `out_loc() -> {dim: label}` — the output cell the file fills. The inherited default works when the coord's fields are all dim names; override otherwise (e.g. a representative ensemble member for a multi-member file). A file holding *only* vertical-group variables (no root var) must include a representative level (e.g. `pressure_level: <a real level>`) so the per-file manifest probe — which resolves `out_loc` to a single chunk — has a concrete level to probe; per-file commit atomicity makes that one level's presence imply the whole file's. The per-ref `out_loc` overrides this representative level with the ref's actual level. (HRRR `wrfprs`/`wrfnat` files are group-only; see `noaa/hrrr/forecast_48_hour_spatial`.)
- `get_index_url() -> str` — *optional*, only if the files have a sidecar byte-range index; used by the obstore-listing discovery util and by index-based `file_refs`. Nothing on the base calls it.

**Optional overrides (working defaults):** `filter_already_present` (manifest probe), `representative_var` (first instant var among the coord's own `data_vars` — only override for a packing that rule misses), the `tick_interval` (1s) / `download_concurrency` (32) class attrs, and `process_virtual_refs` itself (only for a fundamentally different batching policy).

**Discovery utilities** (opt-in, off the base, in `virtual_source_listing.py`): `discover_available_by_obstore_listing(pending, *, store, location_prefix)` works for any obstore backend (S3, GCS, Azure, local) — a file is ready once `store` lists both its data object and its index; the caller passes the built store. A source obstore can't list (an HTTP file server whose directory index is HTML, a frontier that must be probed) gets its own discovery util in the same module and a custom `discover_available`.

## Reader safety: whole files, atomic commits

Every ref from a source file, plus any append-dim expansion its positions require, lands in the same icechunk commit as the rest of that file — a reader on `main` sees either none of a file's data or all of it, never a partially-ingested file. A commit may hold one file (an operational tick) or many (a backfill worker's whole batch); atomicity is per *commit*, and each file is wholly inside one. Operational updates commit per tick, so readers see each lead time's data within seconds of its file publishing.

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

## Operational validation

The `-validate` CronJob runs after each operational update. A dataset lists every check in its `validators()`, mixing two kinds (`validation.DataValidator = XarrayDataValidator | VirtualDataValidator`); `validate_dataset` dispatches by type:

- **`XarrayDataValidator`** — the common kind, `(ds) -> ValidationResult`, run on the opened dataset (e.g. `check_forecast_current_data` for lag). Unchanged across materialized and virtual.
- **`VirtualDataValidator`** — needs manifest/store access, so `validate_dataset` hands it the operational-window region job, the icechunk store, and the opened dataset (each validator uses the subset it needs). The region job is built once (via `operational_update_jobs`) and shared across primary + replica.

Two virtual validators ship in `validation.py`; both are bounded so the whole validation stays well under a couple of minutes even on a large ensemble dataset:

- **`CheckVirtualManifestCompleteness`** — the virtual analog of the materialized `check_for_expected_shards`. Re-runs the operational filter (`region_job.source_file_coords()` + `filter_already_present`) over the recent window and checks, per append-dim position, the fraction of expected source files present against `min_present_fraction` (a tuple indexed newest-first; older positions held to its last value). `(1.0,)` requires every recent position fully ingested; `(0.5, 1.0)` lets the newest be half-published (e.g. GEFS 35-day's slow long lead times) while older positions must be complete; `(0.8,)` requires every position ≥80% (a source that trickles in). Fails — never silently passes — if the window is shorter than the threshold tiers. Reusing the dataset's own coord generation means structural absences (e.g. hour-0 accumulated vars) are already excluded — no false positives. Cheap: one ref-existence probe per file, no decode.
- **`CheckVirtualDecodeHealth`** — where completeness checks references *exist*, this checks the ones that exist actually decode. It keeps only the source files present in the manifest (`filter_already_present`) — so a not-yet-published ref is never mistaken for a decode failure — and decodes a bounded sample (first + last + interior lead times across every ensemble member). `positions="latest"` (default) targets the newest position *with data*, so a broken newest reference is caught at the very next validation rather than a cycle later (when an older `-2` index would otherwise be the one checked); `positions="all"` covers the whole window. A variable fails if any sampled chunk errors or all of its sampled chunks decode entirely NaN; the validator fails — never silently passes — when no references are present. The full-archive NaN/coverage scan stays offline (one full-message decode per chunk across all leads × members is hours).

Tuning (completeness's `min_present_fraction`; decode health's `positions`, `sampled_leads`, `max_workers`) is set where the validator is listed in `validators()` — one place. The materialized `check_for_expected_shards` / `compare_replica_and_primary` do not carry over: virtual stores have no shards, and virtual replicas point at the same source bytes so a value compare adds nothing.

## Multi-group (vertical) datasets

A dataset that mixes single-level variables with variables on a dense vertical dimension is structured as a zarr group hierarchy (see "Vertical levels" in [AGENTS.md](../AGENTS.md) for the user-facing rules). Multi-group is currently exercised only by virtual datasets, so its design lives here.

### Design principles

- **Declarative, group-keyed.** There is one declarative place to find a group's dims, keyed by group; core never hard-codes vertical dim names — each config declares them.
- **Group is a per-variable property, not a job boundary.** A source file can span groups (HRRR `wrfprs` → root single-level *and* `pressure_level`) or be finer than a group (one file per `(var, level)`). Jobs stay scoped as region × source-file var group; a variable's group determines only its zarr path, its dims, and that its refs carry the vertical level.
- **Coordination stays dataset-wide.** Reader-safety (deferred metadata; the icechunk temp-branch merge) is whole-dataset; group structure does not change worker setup/finalize counting.

### Type scheme

In `common/config_models.py`:

- `ROOT` (a `RootGroup` enum sentinel, *not* a `str`) keys the root / single-level group; `DataVar.group: Group = ROOT` is the default.
- `VerticalGroup` is a closed `Literal["pressure_level", "model_level"]` (extend as new dense vertical types are supported); `Group = VerticalGroup | RootGroup`.
- `var_path(group, name)` is the zarr path / identity: `name` at root, else `f"{group}/{name}"`. Because `ROOT` is not a `str` it cannot leak into a path — `ty` narrows `group` to `VerticalGroup` after `if group is ROOT`. A typo'd group key is rejected by both pydantic and `ty`.

A vertical group's name **equals** its dimension name (the `group == dim` invariant): `dims[g]` for a vertical group contains the dim `g`, and there is a `Coordinate(name=g, …)` for it.

### TemplateConfig encoding

- `dims` is group-keyed: `dims: dict[Group, tuple[Dim, ...]]`. A single-level config is `dims = {ROOT: (...)}`; a pressure group is `dims["pressure_level"] = (..., "pressure_level")`. `all_dims` returns the de-duplicated union.
- `DataVar` carries `group`; `var.path = var_path(var.group, var.name)` is its identity. A variable's dims are `dims[var.group]` — never stored on the DataVar, so nothing can disagree.
- `_assert_valid_structure` enforces the invariants (each a self-describing assert): a vertical group adds exactly its own dim to the root dims; every declared group is used and every var's group is declared; `(group, name)` is unique; no root var name collides with a group name; one shared append-dim chunk size across vars; and each var's encoding `chunks`/`shards` length equals `len(dims[var.group])`.

### DataTree representation

The template *is* an `xarray.DataTree` — one node per group (a single-level dataset is a one-node tree, which writes byte-identically to the previous flat-Dataset path). `get_template()` returns the DataTree and `RegionJob.template_ds` is a DataTree.

- `write_metadata` writes the whole tree in one `to_zarr(write_inherited_coords=True)` call. That flag duplicates the shared coords (time/lead time, lat/lon, ensemble_member, spatial_ref) into each child group so a group opens standalone via `open_zarr(group=…)`.
- `DataTree.to_zarr` does not support `append_dim`, so `sync_dims_to` grows the append dim by writing each node's slice with `to_zarr(group=…, append_dim=…)`, sizing each (store, group) from its *own* committed size. Groups may sit at different append-dim lengths transiently under operational lazy growth — acceptable and eventually consistent.
- Virtual refs route by `var.path`: `chunk_key` reads geometry from `template_ds[var.path]` (a vertical-group var resolves to its group node, whose DataArray carries the vertical coord), and `_emit_refs` calls `set_virtual_refs(var.path, …)` on the group-qualified array. A source file spanning groups still commits all its refs together, so per-file atomicity holds across groups.

The materialized chunk-write path is not yet group-aware; a `DynamicalDataset` guard rejects a materialized dataset that declares any non-root variable, so this fails loudly rather than miswriting.

## Storage and reading

A virtual dataset requires every storage config to use the `ICECHUNK` format and an `icechunk_virtual_config` (`IcechunkVirtualConfig` on the `DynamicalDataset`, validated at construction). It holds the real icechunk objects directly — the `VirtualChunkContainer`s registering each source bucket and a `ManifestSplittingConfig` — rather than plain fields, because plain fields would be a lossy subset (no Source Coop S3-compatible endpoint, no GCS mirror, no per-array / multi-dim manifest splits). Nothing serializes the config: workers rebuild the whole dataset from the in-code registry by `dataset_id`, and `StoreFactory` consumes the config directly to register containers and build the (anonymous) `authorize_virtual_chunk_access` map.

### Containers as indirection

A virtual chunk container is more than anonymous credentials — refs are stored *relative* to the registered container prefix:

- **Dedup.** The repeated `s3://bucket/prefix` is not copied into every ref; on an all-virtual store this is most of the on-disk footprint.
- **En-masse repoint.** Swapping a container registration (e.g. NODD-on-AWS → a GCS mirror, or a Source Coop move) repoints every ref at once, with no manifest rewrite.

Container *definitions* are persisted into the repo's config (recovered by `Repository.fetch_config`), so a reader supplies only the (anonymous) credentials map via `authorize_virtual_chunk_access` at open. Credentials are never persisted — a read without them raises.

### Manifest splitting

icechunk manifests are per-array, immutable, and content-addressed; a commit rewrites only the split file(s) whose chunk-index range changed. Each array's manifest is split along the append dimension (`manifest_append_dim_split`), so an operational append lands only in the current (highest-index) split while every historical split is carried over unchanged. **Commit cost is the size of the active split, not the whole array** — this is what keeps the operational ≤ 5 s latency budget real at order-10⁹-ref scale. Size the split by the operational commit-cost ceiling (small enough to rewrite the active split well under a second); for a large ensemble dataset that lands around 1–3 weeks of appends per split.
