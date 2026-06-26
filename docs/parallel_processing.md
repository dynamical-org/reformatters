# Parallel Processing

How reformatters parallelizes work across Kubernetes indexed jobs while ensuring readers always see consistent data.

## Overview

Both backfills and operational updates distribute work across multiple workers using Kubernetes indexed jobs. Each worker independently computes the full list of jobs, then deterministically selects its subset via round-robin assignment — no coordinator or job queue needed.

Work is split along two axes:
- **Regions** — slices along the append dimension (typically one shard each)
- **Variable groups** — subsets of data variables, controlled by `max_vars_per_job`

The Cartesian product of regions and variable groups produces the full job list. Each worker gets every Nth job.

### Append dim region spreading

Regions are reordered with a bit-reversal permutation (`iterating.spread_evenly`) before the job list is built. Round-robin assignment makes worker N's first job region N, so the workers running concurrently (a contiguous index window) would otherwise all hit the same narrow band of the append dim at once. For a multi-year archive that clusters source requests on a few object-store prefixes, hot-spotting partitions that throttle (e.g. S3 503 SlowDown). Spreading the regions makes any contiguous worker window cover the whole append dim, so source load stays even across the run. The permutation is deterministic (every worker recomputes the same order) and concurrency-independent, so it needs nothing beyond the region count.

## The worker-processing seam

`RegionJob.process_worker_jobs(worker_jobs, store_factory, branch_name, worker_index)` is the single polymorphic call the coordinator (`DynamicalDataset._process_region_jobs`) drives every dataset variant through. Each variant owns its store/session lifecycle and commit cadence behind it:

- **Materialized** — opens stores once and writes all of the worker's jobs in a single commit.
- **Virtual** — commits each batch of source-file refs as its generator yields them, because a committed icechunk session is read-only (see [virtual_datasets.md](virtual_datasets.md#the-write-loop)).

The only fork outside this call is the coordination lifecycle: everything runs the parallel temp-branch flow below except virtual operational updates, which are single-writer (see below).

## Reader safety

Readers must always see a consistent view — either the old data or the fully updated data, never a partial state with some variables or time steps missing.

### Structure guard (operational updates)

Before any writes, worker 0 of an operational update asserts that the update template's structure still matches the already-published store — for every variable present in the store, the variable must still exist and its dims, on-disk dtype, chunks, and shards must be unchanged (`template_utils.assert_no_structural_drift_from_existing_store`, called from `DynamicalDataset._process_region_jobs`). A drifted template (a removed/renamed variable or a changed dtype/dims/chunks/shards) would corrupt the existing archive or break readers, so the update fails fast and leaves the live store untouched. Changing structure requires a backfill, which is exempt from this guard.

### Zarr v3 stores

Data chunks can be written directly because they occupy new shard regions that readers won't access until the metadata (which defines the dataset's dimensions) is updated. The metadata write is deferred until the last worker completes, making all new data visible atomically.

For backfills, metadata is written before workers start (the dataset is being created, not read). Specifically, `backfill_local` / `backfill_kubernetes` write metadata to final stores before spawning worker execution. Those calls are required for Zarr v3 support; once the project only supports Icechunk, those calls are no longer needed. `parallel_setup` writes metadata to local tmp storage and to temporary Icechunk branches, but not to final zarr v3 stores. For operational updates, metadata is deferred to finalization.

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

The `from_snapshot_id` check in `reset_branch` prevents two concurrent jobs from both resetting main. The second job to finalize will fail, be retried, and eventually succeed once it picks up the first job's changes.

## Replica ordering

Replicas are always updated before the primary store. This ensures that if a failure occurs between updating replicas and primary, the primary (which drives what work needs to be done) still reflects the pre-update state, causing a retry to redo all the work including re-updating replicas.
