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

For backfills, metadata is written before workers start (the dataset is being created, not read). For operational updates, metadata is deferred to finalization.

### Icechunk stores

All work happens on a temporary branch (`_job_{job_name}`). Readers on `main` are unaffected. The flow:

1. **Worker 0 setup** — creates a temp branch from main's current snapshot, writes expanded metadata, commits on the branch
2. **All workers** — open sessions on the temp branch, write chunk data, commit with `ConflictDetector` rebase (uncooperative distributed writes)
3. **Last worker finalization** — writes final metadata on the branch, then atomically resets `main` to the branch tip using `reset_branch("main", snapshot, from_snapshot_id=original)`. The `from_snapshot_id` check ensures no concurrent process moved main.

## Worker coordination

Workers coordinate via files in an object store directory at `{base_path}/{dataset_id}/_internal/{job_name}/`.

### Setup signal

Worker 0 writes `setup/ready.pkl` after completing setup (creating branches, writing metadata). Workers 1+ poll for this file before proceeding.

### Results

Each worker writes `results/worker-{N}.pkl` containing its `process_results` dict. The last worker (by index) polls until all result files are present, then aggregates them. For updates, the aggregated results drive `update_template_with_results` to trim the template based on what was actually processed.

### Cleanup

After successful finalization, the last worker deletes the `_internal/{job_name}/` directory and the temp icechunk branch.

## Failure modes

### Any worker dies mid-processing

The worker's pod is restarted by Kubernetes. On restart, it re-enters `_process_region_jobs` from the top:
- Reads the existing `setup/ready.pkl` (setup already done by worker 0)
- Opens stores on the same branch (deterministic name)
- Re-processes its jobs (chunk writes are idempotent — icechunk rebase handles conflicts, zarr v3 overwrites)
- Re-writes its results file

Other workers are unaffected.

### Worker 0 dies during setup, restarted

On restart, worker 0 retries setup:
- Branch creation catches "already exists" and reuses the existing branch
- Metadata write is idempotent
- `setup/ready.pkl` is written (or overwritten) when setup completes

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
