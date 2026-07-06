# Parallel icechunk commit contention — options & tradeoffs

Point-in-time evaluation (2026-07-02), following the v0.5.0 HRRR-spatial backfill. The
run completed (389/389 in 2h33m) but exposed a **straggler gridlock** mode; this doc
evaluates the three durable fixes for the next, larger virtual dataset. Tracking: #700.

**Status: not scheduled — we may never need any of this.** Contiguous worker assignment
(#708) and per-group manifest splits (#707) keep commits flat, and backfills complete
comfortably at parallelism ~10. Pick this work up only if a future backfill needs more
multi-process parallelism than that ceiling allows (a much larger dataset, or a
wall-clock requirement parallelism ~10 can't meet).

## The mechanism (measured, instrumented via local repro + `ICECHUNK_LOG` + py-spy)

Uncooperative writers commit via `do_commit_rebasing`: flush → branch-HEAD CAS → on
loss, rebase (cheap) then **re-run the entire flush**. A flush read-modify-writes every
manifest window the changeset touches (~190 objects for a 30-init HRRR-spatial block;
the near-full shared single-level window alone is 143 × ~1.8 MiB ≈ 250 MB down+up), and
manifest ids are random per write, so identical logical content is fully re-merged,
re-compressed, re-uploaded every round (~60s/round near-full, CPU-bound on the
committing thread). **Gridlock condition: round time ≳ competitors' inter-commit
spacing ⇒ CAS win probability collapses.** Measured: workers stuck 30–60+ min across
three fresh sessions; rebase budget is 1000 attempts (~16h at 60s). Fairness never
recovers until competitors drain. p50 commits were fine (1 attempt); the tail was not
(max 85 attempts; 1,672 flushes for 391 commits = 4.3× write amplification, ~1M+
orphaned manifest objects).

## Option 1 — icechunk: reuse unchanged manifest rewrites across rebase rounds

**Spiked and validated** (`../icechunk` branch `spike/rebase-flush-reuse`, +314 lines in
`session.rs` only). Cache per touched window: (sorted parent manifest ids, the window's
modified-chunk table, the written `ManifestRef`+`ManifestFileInfo`). On re-flush, a
window whose four inputs are equal reuses the already-durable object from the prior
round; anything else re-merges normally. Correctness is per-window input equality — no
trust in the conflict solver required (also safe under `BasicConflictSolver`).

- Measured in the spike's forced-rebase test: lost rounds go **8 uploads → 0**
  (downloads and merge CPU skipped too). Production lost round becomes snapshot rebuild
  + tx-log fetch + CAS ≈ ~1–2s → **gridlock cannot form** (round ≪ spacing always).
- Residual cost: a concurrent commit that touches *your* windows (contiguous
  assignment: only the ~6 temporally-adjacent workers sharing a single-level window)
  invalidates those entries → partial re-merge. Bounded and rare.
- Effort: ~1–2 days to productionize (hash fingerprint instead of ChunkTable clone,
  partial-overlap path decision, solver-patching test). No public API surface, no spec
  change. **Upstream-friendly**; bundles with the max_concurrent_nodes + O(M²)-scan
  fixes as one "make many-writer commits scale" contribution.
- Risks: GC could collect a prior round's orphaned manifest mid-retry (same exposure
  class as uncommitted chunks; governed by GC age cutoff — document). Fork-only until
  upstream lands, but backfills already run the fork image.

## Option 2 — commit lease (reformatters-only)

Advisory lease at `_internal/{job_name}/commit-lease` using obstore conditional puts
(**verified available**: `put(mode="create")` raises `AlreadyExistsError`; ETag-
conditioned put gives steal-CAS). Holder heartbeats an expiry; waiters poll with
jitter; stale ⇒ steal via ETag CAS; delete in `finally`. Crash ⇒ bounded stall (TTL,
~60–90s). **Icechunk's CAS remains the correctness guard underneath — a lease bug
degrades to today's behavior, never corruption.**

- Code shape: no mode conditional in shared code. The parallel driver
  (`process_worker_jobs`) constructs the lease and passes it to
  `process_virtual(..., lease=...)`; the single-writer operational path passes nothing
  (no-op default). Materialized `process_worker_jobs` (whose updates are also
  parallel-on-temp-branch) can adopt the same parameter later.
- **Lease-alone does not raise throughput**: it serializes whole commits (flush under
  lease), ceiling = 1/flush-time — the same ceiling as today, minus all waste,
  starvation, and babysitting.
- **Lease + Option 1 combo**: flush *outside* the lease (parallel), hold the lease only
  for rebase+CAS (~1–2s with reuse) → parallel flushes AND zero waste AND strict
  no-starvation. The serialized section shrinks to seconds.
- Effort: ~1 day Python + tests. New distributed-systems surface we own forever
  (heartbeat, steal, clock-skew margins), even if failure-soft.

## Option 3 — cooperative writes (parquet refs + single-writer finalize)

Workers persist refs (e.g. parquet in `_internal/`) and write results files; the
finalizer reads all refs and commits single-writer (batched by window range to bound
memory), then resets `main`. No pickled sessions on a public bucket.

- Eliminates the entire CAS class: **minimum possible manifest churn** (each of ~3,066
  manifests written exactly once vs 1,672 flushes × ~190 = ~300k+ object writes).
- Finalize tail (measured rates): 882M refs at ~14s/2.5M emit ≈ ~80 min single-threaded
  + manifest serialize/upload; parallelizable within the finalizer (emit releases the
  GIL) to ~20–40 min. Needs batched commits by window range (a full-archive changeset
  is tens of GB of memory). Finalizer crash ⇒ restart re-reads parquets (idempotent).
- Strategic upside: unifies materialized + virtual parallel write paths (workers
  produce artifacts; one committer), which is the direction ops/backfill code
  simplification wants long-term.
- Effort: the largest — new artifact format + schema, finalize orchestration, replay
  semantics, memory management. Reformatters-only (no icechunk dependency).

## Recommendation (if the parallelism ceiling ever binds)

1. **Productionize Option 1 first** (~1–2 days): it removes the root cause, is already
   prototyped with a passing forced-rebase test, and upstreams cleanly. Run the next
   backfill on the fork with it; PR upstream alongside the other two icechunk changes.
2. **Hold Option 2** unless the next backfill (with Option 1) still shows tail pain —
   then add it in the combo shape (lease around rebase+CAS only). The seam is designed
   and the primitive verified, so it's a ~day if needed.
3. **Evaluate Option 3 on unification merits, not performance** — schedule it as the
   materialized/virtual write-path consolidation when the next dataset class (or
   operational simplification) justifies it. Performance no longer forces it.

Validation gate for whichever ships next: rebase_attempts p99 ≤ ~5 and zero workers
silent >5 min across a full backfill, at parallelism ≥ 20.
