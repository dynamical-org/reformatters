# Parallel Processing

How reformatters parallelizes work across Kubernetes indexed jobs while ensuring readers always see consistent data.

## Overview

Both backfills and operational updates distribute work across multiple workers using Kubernetes indexed jobs. Each worker independently computes the full list of jobs, then deterministically selects its subset via round-robin assignment — no coordinator or job queue needed.

Work is split along two axes:
- **Regions** — slices along the append dimension (typically one shard each)
- **Variable groups** — subsets of data variables, controlled by `max_vars_per_job`

The Cartesian product of regions and variable groups produces the full job list. Each worker gets every Nth job.

## Reader safety

Readers must always see a consistent view — either the old data or the fully updated data, never a partial state with some variables or time steps missing.

### Zarr v3 stores

Data chunks can be written directly because they occupy new shard regions that readers won't access until the metadata (which defines the dataset's dimensions) is updated. The metadata write is deferred until the last worker completes, making all new data visible atomically.

For backfills, metadata is written before workers start (the dataset is being created, not read). Specifically, `backfill_local` / `backfill_kubernetes` write metadata to final stores before spawning worker execution. Those calls are required for Zarr v3 support; once the project only supports Icechunk, those calls are no longer needed. `parallel_setup` writes metadata to local tmp storage and to temporary Icechunk branches, but not to final zarr v3 stores. For operational updates, metadata is deferred to finalization.

### Icechunk stores

All metadata and chunk writes happen on a temporary branch (`_job_{job_name}`). Readers on `main` are unaffected. The flow:

1. **Worker 0 setup** — creates a temp branch from main's current snapshot, copies expanded metadata from the local tmp store, **commits** on the branch. This must be a commit (not amend) so the original main snapshot is preserved as the parent of the new work.
2. **All workers** — open sessions on the temp branch, write chunk data, **amend** the temp branch tip (via `amend_if_icechunk`, which falls back to `rebase` + `amend` on `ConflictError`). Each amend replaces the previous tip, so per-worker writes collapse into a single snapshot rather than each becoming its own snapshot.
3. **Last worker finalization** — if `update_template_with_results` produced a template that differs from the input (e.g. the time dim was trimmed or a coordinate or attribute was updated), writes the new metadata on the branch and **amends** it into the existing tip (via `amend_if_icechunk`). If the template is unchanged, the temp tip already has the right metadata, so the rewrite + amend is skipped. Either way, finalize then atomically resets `main` to the temp branch tip using `reset_branch("main", snapshot, from_snapshot_id=original)`. This branch reset is what makes all writes visible to readers. The `from_snapshot_id` check ensures no concurrent process moved main.

The net effect on `main`'s history is **typically** exactly one new snapshot per backfill or operational update — the per-worker and intermediate-metadata writes never appear as separate snapshots. The temp branch itself is deleted after finalization.

The one exception is worker 0 setup retry: setup is skipped wholesale on restart only if `setup/ready.json` was written (which happens at the very end of setup). If worker 0 dies between the "Expand dataset" commit and writing `ready.json`, the next attempt will re-run setup and add another "Expand dataset" commit on the temp branch; subsequent worker amends keep that commit as their parent, so main's ancestry gains an extra snapshot per such retry. The same applies in single-worker runs because `ready.json` is only written when `workers_total > 1`. In all cases the data and reader-safety guarantees are unchanged — there are just one or more extra "Expand dataset" snapshots in main's history.

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
- Re-processes its jobs (chunk writes are idempotent — icechunk amend rebases on conflict, zarr v3 overwrites)
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
