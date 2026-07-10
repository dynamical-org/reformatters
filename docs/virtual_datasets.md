# Virtual Datasets

A virtual dataset is an Icechunk store whose chunks are references — `(location, offset, length)` pointers into source files (e.g. GRIB messages) on the provider's object store — rather than copied bytes. A per-variable zarr serializer (e.g. `GribberishCodec`) decodes the referenced bytes at read time. Virtual datasets are written by `VirtualRegionJob` subclasses (`src/reformatters/common/virtual_region_job.py`).

How virtual jobs plug into worker parallelism and coordination is covered in [parallel_processing.md](parallel_processing.md); this doc covers what is specific to writing and reading virtual datasets.

## Why virtual

A virtual dataset complements the materialized (rechunked) time-series dataset over the same source data:

- **Spatial / map-optimized chunking.** Chunks follow the native source-file layout — usually one time step over the full spatial grid — so a virtual `-spatial` dataset serves map and spatial queries efficiently, while the materialized dataset provides the complementary time-series chunking.
- **Every curated source variable, cheaply.** A variable costs only a handful of byte-range refs, not a rechunked copy, so a virtual dataset can include every variable we have metadata for rather than just the materialized subset.
- **Very low latency updates.** Writing refs records byte offsets without moving data, so a new source file's data is visible within seconds of publication (target ≤ 5 s end to end). See [Operational updates](#operational-updates-single-writer).

## Scale

Virtual datasets are *metadata-heavy*, not storage-heavy. One GRIB message — one `(variable, …, lead/time)` field — becomes one virtual chunk, so ref counts track the full Cartesian product of the dataset's dimensions *including the variable axis*. A multi-year ensemble dataset reaches order 10⁹ refs and tens of millions of source index files. Two consequences shape the design:

- **Manifests must be split** (see [Manifest splitting](#manifest-splitting)) — a single per-array manifest at this ref count is too large to rewrite on every commit.
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
- `out_loc() -> {dim: label}` — the output cell the file fills. The inherited default works when the coord's fields are all dim names; override otherwise (e.g. a representative ensemble member for a multi-member file). A file holding *only* vertical-group variables (no root var) must include a representative level (e.g. `pressure_level: <a real level>`) so the per-file manifest probe — which resolves `out_loc` to a single chunk — has a concrete level to probe; per-file commit atomicity makes that one level's presence imply the whole file's. The per-ref `out_loc` overrides this representative level with the ref's actual level. (HRRR `wrfprs`/`wrfnat` files are group-only.)
- `get_index_url() -> str` — *optional*, only if the files have a sidecar byte-range index; used by the obstore-listing discovery util and by index-based `file_refs`. Nothing on the base calls it.

**Optional overrides (working defaults):** `filter_already_present` (manifest probe), `representative_var` (first instant var among the coord's own `data_vars` — only override for a packing that rule misses), the `tick_interval` (1s) / `download_concurrency` (32) class attrs, and `process_virtual_refs` itself (only for a fundamentally different batching policy).

**Discovery utilities** (opt-in, off the base, in `virtual_source_listing.py`): `discover_available_by_obstore_listing(pending, *, store, location_prefix)` works for any obstore backend (S3, GCS, Azure, local) — a file is ready once `store` lists both its data object and its index; the caller passes the built store. A source obstore can't list (an HTTP file server whose directory index is HTML, a frontier that must be probed) gets its own discovery util in the same module and a custom `discover_available`.

## Reader safety: whole files, atomic commits

Every ref from a source file, plus any append-dim expansion its positions require, lands in the same icechunk commit as the rest of that file — a reader on `main` sees either none of a file's data or all of it, never a partially-ingested file. A commit may hold one file (an operational tick) or many (a backfill worker's whole batch); atomicity is per *commit*, and each file is wholly inside one. Operational updates commit per tick, so readers see each lead time's data within seconds of its file publishing.

This invariant is also what lets the filter probe a single representative cell per file: if one cell a file covers is present, the whole file is present.

## Operational updates: single writer

`operational_update_jobs()` returns **one** job spanning the active window (the recent, still-incomplete span of the append dim). One pod commits whole files straight to `main` as they arrive — no temp branch, no coordination files (`DynamicalDataset.update` routes to `_run_virtual_operational_update`).

Single-writer is what makes lazy append-dim expansion safe: `main` grows exactly to the data ingested (never NaN-padded toward "now"), and a resize can never race another writer. If an update falls far behind, run a backfill to catch up rather than scaling operational updates to multiple workers.

Each update fire first refreshes metadata from the checked-in template, trimmed to each group's committed extent (`VirtualRegionJob.refresh_metadata`): template attrs and coordinate-value fixes deploy operationally — matching materialized updates, which rewrite metadata every fire — while the append dim is never sized past ingested data. Byte-identical metadata is skipped, so an in-sync store adds no commit. Structural changes are rejected by the same drift guard materialized updates use — changing structure still requires a backfill.

Virtual stores carry no consolidated metadata in the root zarr.json — nothing updates it as the append dim grows, so readers list the live metadata instead. Every virtual metadata writer renders unconsolidated (`VirtualRegionJob.consolidated_metadata = False` covers backfill setup/finalize; appends and the refresh pass `consolidated=False` directly), so their outputs are byte-identical and the refresh only commits on real template drift. Appends write only variables spanning the append dim: overwriting the rest would delete chunks that equal their fill_value (e.g. `spatial_ref`), since appends write with `write_empty_chunks=False`.

Crash recovery is automatic: committed refs are durable and the filter skips them on the next cron fire; uncommitted refs and expansions vanish and are re-discovered.

## Backfill: parallel on a pre-sized temp branch

Virtual backfills use the standard parallel temp-branch flow (see [parallel_processing.md](parallel_processing.md#icechunk-stores)). Worker 0 pre-sizes the full template on the branch, so every worker's `sync_dims_to` is a no-op and parallel workers write disjoint refs with no resize conflicts. Jobs partition by chunks along the append dim (virtual arrays have no shards). Workers are assigned contiguous append-dim blocks of jobs (`worker_assignment = "contiguous"`), so each flush rewrites only the manifest windows its own block covers rather than most windows of every array (see [parallel_processing.md](parallel_processing.md#append-dim-region-spreading-and-worker-assignment)).

Commit latency ≈ `(1 + rebase_attempts) × flush cost`. A flush read-modify-writes every manifest window the session's refs touch (manifests are immutable), and every lost branch-HEAD CAS race re-runs the flush, so rebase attempts scale with parallelism. Keeping each worker's refs within its own few windows — contiguous append-dim assignment — is what bounds flush cost: measured on the HRRR-virtual backfill, commits held flat (~10–30s at parallelism 10) from empty to full archive, where scattered (spread) assignment touched most windows of every array per flush and commits grew ~40s → 1000s+ as windows filled. Icechunk stamps `rebase_attempts` into each snapshot's metadata (`repo.ancestry`) — read it to distinguish contention from flush cost when commits are slow.

Backfill parallelism has a low ceiling: measured on HRRR-virtual, throughput peaked by ~10 concurrent workers, and doubling to 20 added contention without adding throughput. The failure mode is gridlock, not a slow mean: once a worker's flush round takes longer than its competitors' inter-commit spacing, it almost never wins the branch-HEAD race and re-flushes indefinitely — most commits land within a few rebase attempts while a stuck few spiral until competitors drain. Watch the max `rebase_attempts`, not just the mean. Deleting a stuck worker's pod is safe: nothing it wrote is durable before its commit, and the replacement worker re-filters and resumes. If a backfill ever needs more parallelism than this ceiling allows, options to raise it (with spikes and tradeoffs) are evaluated in [docs/plans/virtual_commit_contention_options.md](plans/virtual_commit_contention_options.md).

Two operational rules when backfilling a live virtual dataset:

- **Suspend the dataset's `-update` CronJob for the duration of the backfill.** Finalize resets `main` to the temp branch only if `main` hasn't moved since setup; an operational fire committing to `main` mid-backfill would make finalize skip the reset (with a warning), discarding the backfill's work.
- **Choose `append_dim_end` as the last *fully published* position, not "now".** Finalize resets `main` to the pre-sized branch, so positions past the published data would appear as NaN-filled slots to readers.

## Manifest splitting

Icechunk stores each array's chunk references in one or more immutable manifest objects, split along a dimension via `IcechunkVirtualConfig.manifest_split` (built with `manifest_append_dim_split`). A commit rewrites only the split(s) whose refs changed: an operational append lands in the current (highest-index) split of each touched array while historical splits carry over unchanged, so operational commit cost is the active splits' size, not the archive's. Two independent costs pull in opposite directions when sizing splits:

- **Manifest size (bytes)** — a reader downloads a whole manifest to resolve *any* chunk in it, and a commit rewrites the active split's whole manifest each time it grows. Keep a full manifest within a reader-friendly size: rough budgets are ≤ 3 MiB per single-level variable (a small web app plotting one field pays one manifest + one chunk) and 5–8 MiB for vertical-group variables (those readers expect bulk downloads). Larger split → bigger manifests. A manifest's size is `split_size × refs_per_index × bytes_per_ref`, so arrays with more refs per append index (more levels, ensemble members, lead times) reach a given size at a smaller `split_size`.
- **Total manifest count `M`** — `M = array_count × ceil(appends / split_size)`. Every commit's cost grows with `M` (it re-serializes the snapshot's manifest list and touches every split of every array). Smaller split → more manifests → larger `M`. For a many-variable dataset `M` is dominated by `array_count`, so over-splitting the low-ref-count arrays is the biggest, least useful contributor.

These two are set per array, so size them per array group: give low-ref-count arrays (e.g. single-level) a **coarse** split (few splits, each manifest still small) and high-ref-count arrays (e.g. vertical groups) a **finer** split (more splits, but each manifest stays within the reader budget). `manifest_append_dim_split` takes either one `split_size` for all arrays or a `{path_regex: size, None: catch_all_size}` mapping for this per-group policy.

To convert a size budget into a `split_size` you need **bytes per ref**. Measured on HRRR-virtual: **~16.4 bytes/ref** (weighted; median ~17). This is a rough number from a single dataset — location compressibility varies with URL structure, so **check it on each new dataset** once some data is written: `repo.list_manifest_files(snapshot_id)` returns every live manifest's exact `size_bytes` and `num_chunk_refs`. Don't estimate from S3 object sizes (partially-filled splits skew low, and you can't tell fullness from the outside). Two gotchas: icechunk's zstd location compression only engages at ≥ `min_num_chunks` refs per manifest (default 1000) — below that, raw ~80-byte URLs are stored, so keep every group's split above ~1000 refs; and refs are only this small when a `VirtualChunkContainer` prefix covers the locations.

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

## Storage and reading

A virtual dataset requires every storage config to use the `ICECHUNK` format and an `icechunk_virtual_config` (`IcechunkVirtualConfig` on the `DynamicalDataset`, validated at construction). It holds the real icechunk objects directly — the `VirtualChunkContainer`s registering each source bucket and a `ManifestSplittingConfig` — rather than plain fields, because plain fields would be a lossy subset (no Source Coop S3-compatible endpoint, no GCS mirror, no per-array / multi-dim manifest splits). Nothing serializes the config: workers rebuild the whole dataset from the in-code registry by `dataset_id`, and `StoreFactory` consumes the config directly to register containers and build the (anonymous) `authorize_virtual_chunk_access` map.

### Containers as indirection

A virtual chunk container is more than anonymous credentials — refs are stored *relative* to the registered container prefix:

- **Dedup.** The repeated `s3://bucket/prefix` is not copied into every ref; on an all-virtual store this is most of the on-disk footprint.
- **En-masse repoint.** Swapping a container registration (e.g. NODD-on-AWS → a GCS mirror, or a Source Coop move) repoints every ref at once, with no manifest rewrite.

Container *definitions* are persisted into the repo's config (recovered by `Repository.fetch_config`), so a reader supplies only the (anonymous) credentials map via `authorize_virtual_chunk_access` at open. Credentials are never persisted — a read without them raises.

How manifests are split, and how to size the splits, is covered in [Manifest splitting](#manifest-splitting).
