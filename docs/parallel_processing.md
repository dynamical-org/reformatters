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

The only fork outside this call is the coordination lifecycle: extent-extending jobs run the parallel temp-branch flow below, extent-preserving overwrite backfills run the cooperative publish, and virtual operational updates are single-writer (see below).

## Reader safety

Readers must always see a consistent view — either the old data or the fully updated data, never a partial state with some variables or time steps missing.

### Structure guard (operational updates)

Before any writes, worker 0 of an operational update asserts that the update template's structure still matches the already-published store — for every variable present in the store, the variable must still exist and its dims, on-disk dtype, chunks, and shards must be unchanged (`template_utils.assert_no_structural_drift_from_existing_store`). A drifted template (a removed/renamed variable or a changed dtype/dims/chunks/shards) would corrupt the existing archive or break readers, so the update fails fast and leaves the live store untouched. Changing structure requires a backfill.

### Overwrite guard (backfills into an existing store)

A backfill into an existing store (`--overwrite-chunks` / `--overwrite-metadata`) runs `template_utils.assert_safe_overwrite` — in the `backfill-kubernetes` driver before submitting, and again on worker 0 before any writes (the deployed image's template can differ from the driver's). It rejects structural drift of arrays the store already has, any template shorter than the store along the append dim (trimming an existing store is never supported), new arrays unless `--overwrite-metadata` was passed, and a longer template unless an explicit `--append-dim-end` was given with both overwrite flags. Overwrite metadata writes also exclude coordinate value chunks the template renders entirely null (`template_utils.store_written_coords`, e.g. `ingested_forecast_length`) so job-written coordinate state is never clobbered by the template's empty values.

### Zarr v3 stores

Data chunks can be written directly because they occupy new shard regions that readers won't access until the metadata (which defines the dataset's dimensions) is updated. The metadata write is deferred until the last worker completes, making all new data visible atomically.

For fresh-store backfills, metadata is written before workers start (the dataset is being created, not read). Specifically, `backfill_local` / `backfill_kubernetes` write metadata to final stores before spawning worker execution. `parallel_setup` writes metadata to local tmp storage and to temporary Icechunk branches, but not to final zarr v3 stores. For operational updates and expansion backfills, metadata is deferred to finalization, so an extension appears only after all chunk data is written. Extent-preserving overwrite backfills instead refresh metadata at setup (see "Cooperative backfill publish"): a new variable appears immediately, all-NaN, and fills in as chunks land.

### Icechunk stores

Extent-extending jobs — operational updates, expansion backfills, and new-store backfills — do all metadata and chunk writes on a temporary branch (`_job_{job_name}`). Readers on `main` are unaffected. The flow:

1. **Worker 0 setup** — creates a temp branch from main's current snapshot, copies expanded metadata from the local tmp store, commits on the branch
2. **All workers** — open sessions on the temp branch, write chunk data, commit with `ConflictDetector` rebase (uncooperative distributed writes)
3. **Last worker finalization** — writes final metadata on the branch, then atomically resets `main` to the branch tip using `reset_branch("main", snapshot, from_snapshot_id=original)`. This branch reset is what makes all writes visible to readers. The `from_snapshot_id` check ensures no concurrent process moved main.

Extent-preserving overwrite backfills use the cooperative publish below instead, so operational updates can keep running while they do.

### Cooperative backfill publish (overwrite backfills)

An overwrite backfill that does not extend the store (a new variable, a re-backfill of flagged positions) writes into the live store with no temp branch and no `reset_branch`, so it can run for days while operational updates keep publishing:

1. **Worker 0 setup** — refreshes every store's metadata from the template via `refresh_store_metadata`: trimmed to the store's extent at that moment (never resizing arrays the updates are growing), creating any newly added arrays. A new variable is therefore reader-visible immediately, all-NaN, and fills in as the backfill and the updates write it — the same progressive visibility an operational update gives a new variable.
2. **All workers** — write chunk data through icechunk *fork sessions* (`Session.fork()`): chunk bytes upload to object storage as they are written, while the chunk refs accumulate in the fork's changeset. Each worker stores its serialized changeset as a coordination file (`sessions/{role}/worker-{N}`) instead of committing. Zarr v3 stores are written directly, as always.
3. **Last worker finalization** — per repo (replicas first), merges every worker's changeset into one fresh session on `main` and commits once, rebasing over anything committed while the backfill ran. Chunk conflicts resolve in the other job's favor (`VersionSelection.UseTheirs` — the operational update wins). A commit that landed mid-flush and changed array metadata (an update extending the append dim) is unresolvable inside icechunk's rebase, so the finalizer rebuilds the session from the new tip — which then already contains that commit — re-merges, and retries. A `finalized/{role}` marker makes a finalize retry after a crash skip repos that already committed.

Because the backfill's chunk refs live only in coordination files until the final commit, do not run `garbage-collect` (src/scripts/icechunk_utils.py) with a cutoff newer than the start of any in-flight backfill — the uploaded-but-unreferenced chunks would be collected.

Virtual backfills follow the same shape with one difference: their refs would make impractically large changesets, so each worker commits its batches straight to `main` (whole source files per commit keeps readers safe, exactly like virtual operational updates) with the same conflict-solver + fresh-session retry.

### Virtual Icechunk operational updates (single-writer exception)

Virtual Icechunk datasets (`VirtualRegionJob`) are the one exception to the temp-branch coordinator. Their *operational* updates run **single-writer** and commit whole source files straight to `main` as each arrives, so readers see new data within seconds rather than at finalization. There is no temp branch, no `parallel_setup`, no coordination files, and no finalization step — `update()` routes virtual operational updates to `_run_virtual_operational_update` instead of `_process_region_jobs`.

This is safe because each commit contains a *whole* source file's references (all of its chunks), so a reader on `main` always sees either none or all of a file's data. Reader-visible atomicity comes from icechunk's per-commit transaction, not from a branch swap. See [virtual_datasets.md](virtual_datasets.md).

Virtual *backfills* are **not** an exception — they use the normal temp-branch coordinator above (parallel across workers, pre-sized branch, finalize resets `main`).

## Worker coordination

For `workers_total > 1`, workers coordinate via files in an object store directory at `{base_path}/{dataset_id}/_internal/{job_name}/` (a single-worker job skips these files).

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
- **Died after resetting some replicas but not primary** — replicas are ahead of primary. On retry, finalize detects a repo whose main is already on the temp branch (reset by the previous attempt) and skips it, then resets the remaining repos, primary last.
- **Died after resetting all stores but before branch cleanup** — data is fully committed. Orphan branch and coordination files remain but don't affect correctness. A fresh job uses a different job name.

In all cases, `main` either hasn't moved (safe) or has moved to the correct final state. Reader-visible data is never corrupted.

### Worker exhausts per-index retry limit

The entire Kubernetes job fails. The team is notified and can run a fresh job. Since the fresh job has a different job name, it gets a clean `_internal/` namespace and a new branch — no interference from the failed run.

### Concurrent jobs writing to the same dataset

Overwrite backfills and operational updates are designed to run concurrently; where they would conflict, the update wins:

- **Disjoint regions by construction.** The backfill driver caps the job list at the position the update jobs currently start from (`--filter-end`, computed once so every worker sees the same list). Updates rewrite everything at or after that position — including any newly added variable — on every fire, and their start only moves forward in time, so neither job writes the other's chunks. Pass `--no-defer-to-updates` only when the update cron is suspended.
- **Publish without displacing.** The cooperative publish above commits the backfill onto whatever `main` has become; an update publishing first is preserved (nothing is reset), and residual chunk conflicts resolve in the update's favor.
- **The one residual collision.** A materialized update's own `reset_branch` CAS fails if the backfill's (at most two) `main` commits — the metadata refresh at setup and the merged commit at finalize — land during that update's run. That update run fails loudly with "main moved during this job", leaving its temp branch and coordination files for inspection; the next cron fire redoes its work from the new main. Expect at most a couple of these benign failures over a multi-day backfill; the orphaned `_job_*` branch can be deleted.

Two *extent-extending* jobs (an update and an expansion backfill, or two updates) must still not overlap: both publish via `reset_branch`, whose `from_snapshot_id` check makes the second to finalize fail, and icechunk cannot rebase a committed temp branch. The job that moved main wins and the loser's work must be re-run.

## Replica ordering

Replicas are always updated before the primary store. This ensures that if a failure occurs between updating replicas and primary, the primary (which drives what work needs to be done) still reflects the pre-update state, causing a retry to redo all the work including re-updating replicas.
