# Parallel Processing

How reformatters parallelizes work across Kubernetes indexed jobs while ensuring readers always see consistent data.

## Overview

Both backfills and operational updates distribute work across multiple workers using Kubernetes indexed jobs. Each worker independently computes the full list of jobs, then deterministically selects its subset per its region job class's `worker_assignment` policy — no coordinator or job queue needed.

Work is split along two axes:
- **Regions** — slices along the append dimension (typically one shard each)
- **Variable groups** — subsets of data variables, controlled by `max_vars_per_job`

The Cartesian product of regions and variable groups produces the full job list, in canonical append-dim order. `iterating.get_worker_jobs` partitions it per the region job class's `worker_assignment`, each mode owning its own ordering + selection: the default `"spread"` permutes the list with `spread_evenly` then gives each worker every Nth job; virtual region jobs use `"contiguous"` and each worker gets one contiguous block in list order.

### Append dim region spreading and worker assignment

**Materialized (`worker_assignment = "spread"`).** Worker assignment reorders the append-dim-ordered job list with a bit-reversal permutation (`iterating.spread_evenly`) before round-robin selection. Round-robin over the unpermuted list would make worker N's first job region N, so the workers running concurrently (a contiguous index window) would all hit the same narrow band of the append dim at once. For a multi-year archive that clusters source requests on a few object-store prefixes, hot-spotting partitions that throttle (e.g. S3 503 SlowDown). Spreading the jobs makes any contiguous worker window cover the whole append dim, so source load stays even across the run. The permutation is deterministic (every worker recomputes the same order) and concurrency-independent, so it needs nothing beyond the job count.

**Virtual (`worker_assignment = "contiguous"`).** Virtual worker assignment keeps the job list in append-dim order and gives each worker one contiguous block. Icechunk splits each array's manifest into windows along the append dim, and manifests are immutable — a commit read-modify-writes every window its refs touch. A worker whose regions are scattered across the whole append dim touches most windows of every array on every flush (thousands of manifest rewrites, with bytes growing as the archive fills); a contiguous block touches only 1-2 windows per array and rewrite bytes stay bounded by one window. Contiguous assignment concentrates .idx reads and prefix listings on adjacent source prefixes, which can still hot-spot; aligning workers with the manifest split structure matters more, since virtual workers fetch only small index files rather than the data files.

## The worker-processing seam

`RegionJob.process_worker_jobs(worker_jobs, store_factory, branch_name, worker_index)` is the single polymorphic call the coordinator (`DynamicalDataset._process_region_jobs`) drives every dataset variant through. Each variant owns its store/session lifecycle and commit cadence behind it:

- **Materialized** — opens stores once and writes all of the worker's jobs in a single commit.
- **Virtual** — gathers the worker's not-already-present source files across all its jobs, then commits each batch its generator yields (a backfill yields once → one commit per worker, like materialized; an operational update yields per poll tick), because a committed icechunk session is read-only (see [virtual_datasets.md](virtual_datasets.md#the-write-loop)).

The only fork outside this call is the coordination lifecycle: everything runs the parallel temp-branch flow below except virtual operational updates, which are single-writer (see below).

## Reader safety

Readers must always see a consistent view — either the old data or the fully updated data, never a partial state with some variables or time steps missing.

### Structure guard (operational updates)

Before any writes, worker 0 of an operational update asserts that the update template's structure still matches the already-published store — for every variable present in the store, the variable must still exist and its dims, on-disk dtype, chunks, and shards must be unchanged (`template_utils.assert_no_structural_drift_from_existing_store`, called from `DynamicalDataset._process_region_jobs`). A drifted template (a removed/renamed variable or a changed dtype/dims/chunks/shards) would corrupt the existing archive or break readers, so the update fails fast and leaves the live store untouched. Changing structure requires a backfill.

### Overwrite guard (backfills into an existing store)

A backfill into an existing store (`--overwrite-chunks` / `--overwrite-metadata`) runs `template_utils.assert_safe_overwrite` — in the `backfill-kubernetes` driver before submitting, and again on worker 0 before any writes (the deployed image's template can differ from the driver's). It rejects structural drift of arrays the store already has, any template shorter than the store along the append dim (trimming an existing store is never supported), new arrays unless `--overwrite-metadata` was passed, and a longer template unless an explicit `--append-dim-end` was given with both overwrite flags. Overwrite metadata writes also exclude coordinate value chunks the template renders entirely null (`template_utils.store_written_coords`, e.g. `ingested_forecast_length`) so job-written coordinate state is never clobbered by the template's empty values.

### Zarr v3 stores

Data chunks can be written directly because they occupy new shard regions that readers won't access until the metadata (which defines the dataset's dimensions) is updated. The metadata write is deferred until the last worker completes, making all new data visible atomically.

For fresh-store backfills, metadata is written before workers start (the dataset is being created, not read). Specifically, `backfill_local` / `backfill_kubernetes` write metadata to final stores before spawning worker execution. Those calls are required for Zarr v3 support; once the project only supports Icechunk, those calls are no longer needed. `parallel_setup` writes metadata to local tmp storage and to temporary Icechunk branches, but not to final zarr v3 stores. For operational updates and overwrite backfills, metadata is deferred to finalization, so a metadata change (a new variable, an extension) appears only after all chunk data is written.

### Icechunk stores

All metadata and chunk writes happen on a temporary branch (`_job_{job_name}`). Readers on `main` are unaffected. The flow:

1. **Worker 0 setup** — creates a temp branch from main's current snapshot, copies expanded metadata from the local tmp store, commits on the branch
2. **All workers** — open sessions on the temp branch, write chunk data, commit with `ConflictDetector` rebase (uncooperative distributed writes)
3. **Last worker finalization** — writes final metadata on the branch, then atomically resets `main` to the branch tip using `reset_branch("main", snapshot, from_snapshot_id=original)`. This branch reset is what makes all writes visible to readers. The `from_snapshot_id` check ensures no concurrent process moved main.

### Virtual Icechunk operational updates (single-writer exception)

Virtual Icechunk datasets (`VirtualRegionJob`) are the one exception to the temp-branch coordinator. Their *operational* updates run **single-writer** and commit whole source files straight to `main` as each arrives, so readers see new data within seconds rather than at finalization. There is no temp branch, no `parallel_setup`, no coordination files, and no finalization step — `update()` routes virtual operational updates to `_run_virtual_operational_update` instead of `_process_region_jobs`.

This is safe because each commit contains a *whole* source file's references (all of its chunks), so a reader on `main` always sees either none or all of a file's data. Reader-visible atomicity comes from icechunk's per-commit transaction, not from a branch swap. See [virtual_datasets.md](virtual_datasets.md).

Virtual *backfills* are **not** an exception — they use the normal temp-branch coordinator above (parallel across workers, pre-sized branch, finalize resets `main`).

## Worker coordination

Workers coordinate via files in an object store directory at `{base_path}/{dataset_id}/_internal/{job_name}/`.

### Setup signal

Worker 0 writes `setup/ready.json` after completing setup (creating branches, writing metadata). Workers 1+ poll for this file before proceeding.

### Results

Each worker writes `results/worker-{N}.json` containing its `process_results` dict. The last worker (by index) polls until all result files are present, then aggregates them. For updates, the aggregated results drive `update_template_with_results` to trim the template based on what was actually processed.

### Cleanup

After successful finalization, the last worker deletes the `_internal/{job_name}/` directory and the temp icechunk branch.

## Failure modes

### Any worker dies mid-processing

The worker's pod is restarted by Kubernetes. On restart, it re-enters `_process_region_jobs` from the top:
- Reads the existing `setup/ready.json` (setup already done by worker 0)
- Opens stores on the same branch (deterministic name)
- Re-processes its jobs (chunk writes are idempotent — icechunk rebase handles conflicts, zarr v3 overwrites)
- Re-writes its results file

Other workers are unaffected.

### Worker 0 dies during setup, restarted

On restart, worker 0 retries setup:
- Branch creation catches "already exists" and reuses the existing branch
- Metadata write is idempotent
- `setup/ready.json` is written (or overwritten) when setup completes

Workers 1+ that were polling for setup will proceed once the file appears.

### Last worker dies during finalization

Finalization is not atomic. Possible partial states:
- **Died before any `reset_branch`** — main unchanged, all data is on the temp branch. Retry re-enters finalization and completes it.
- **Died after resetting some replicas but not primary** — replicas are ahead of primary. Primary drives future work, so a retry or fresh job will redo everything. Replicas get re-updated (idempotent), then primary gets updated.
- **Died after resetting all stores but before branch cleanup** — data is fully committed. Orphan branch and coordination files remain but don't affect correctness. A fresh job uses a different job name.

In all cases, `main` either hasn't moved (safe) or has moved to the correct final state. Reader-visible data is never corrupted.

### Worker exhausts per-index retry limit

The entire Kubernetes job fails. The team is notified and can run a fresh job. Since the fresh job has a different job name, it gets a clean `_internal/` namespace and a new branch — no interference from the failed run.

### Concurrent jobs writing to the same dataset

The `from_snapshot_id` check in `reset_branch` prevents two concurrent jobs from both resetting main. Icechunk can rebase only uncommitted sessions, not a committed temp branch, so a job that finds main moved past its starting snapshot cannot merge — finalize raises, leaving its temp branch and coordination files in place for inspection, and the job that moved main wins. (A finalize retry after a crash is distinguished by checking whether main's snapshot is already on the temp branch.) The losing job's work must be re-run.

Because any operational update that publishes during an overwrite backfill makes the backfill lose this way, time overwrite backfills to avoid update publishes — for a long backfill on a frequently-updating dataset, manually suspend the update cronjob for the duration (`kubectl patch cronjob <name> -p '{"spec":{"suspend":true}}'`; a deploy to main re-applies cronjobs and resumes it).

## Replica ordering

Replicas are always updated before the primary store. This ensures that if a failure occurs between updating replicas and primary, the primary (which drives what work needs to be done) still reflects the pre-update state, causing a retry to redo all the work including re-updating replicas.
