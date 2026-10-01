# Backfill

Populate a dataset's store — create and fill a brand-new store, add a newly-implemented variable, or re-run the positions a validation pass flagged. Backfills distribute work across Kubernetes indexed jobs the same way operational updates do; see [parallel_processing.md](parallel_processing.md) for how workers coordinate.

Run a backfill only after the dataset's code is merged to `main`: the GitHub action and the deployed container image both build from `main`, so this keeps the driver and workers on the same commit.

## Prerequisites (new dataset)

- Create the public bucket once: `./deploy/aws/create_new_aws_open_data_bucket.sh <provider>-<model>`.
- Contact feedback@dynamical.org if you are setting up new compute or storage.

## Where to run

- **GitHub Action (preferred).** The [Manual: Backfill](https://github.com/dynamical-org/reformatters/actions/workflows/manual-backfill.yml) action (workflow_dispatch, requires repo write access) runs only from `main`, waits for main's tip to finish deploying, and submits the job with that deploy's image. It exposes only the safe operations.
- **Kubernetes from your machine.** `DYNAMICAL_ENV=prod uv run main <dataset-id> backfill-kubernetes [flags]`, then track with `kubectl get jobs`. Complete README.md > Deploying to the cloud > Setup first.

## Operations

Pick the operation by what you're doing (action operation name / equivalent CLI flags):

- **New store** — `create-new-store` / `backfill-kubernetes`. Creates the store and fails if one already exists. `append_dim_end` defaults to now (leave it empty to backfill through now).
- **New variable** — `overwrite-chunks-and-metadata` / `backfill-kubernetes --overwrite-chunks --overwrite-metadata --filter-variable-names <name>`. Refreshes metadata from the template (creating the variable; the guards never trim the store) and writes its chunk data. The store extent is unchanged unless you set an `append_dim_end` past the current end.
- **Refresh metadata only** (an attribute change, no data rewrite) — `overwrite-metadata` / `backfill-kubernetes --overwrite-metadata`. Rewrites metadata in place; launches no workers. Use for attribute changes only: changing encoding this way reinterprets the existing on-disk bytes and can break readers, so make encoding changes only with extreme care (and a full chunk rewrite).
- **Rewrite or re-backfill chunk data** — `overwrite-chunks` / `backfill-kubernetes --overwrite-chunks [--filter-...]`. For specific flagged positions, `--filter-contains` (repeatable — pass it once per append-dim timestamp) is the most efficient: it runs only the region jobs those timestamps touch, rather than the whole `filter_start`/`filter_end` window. The validation `availability` scan lists the flagged timestamps in `unavailable_timestamps.txt`.

`uv run main <dataset-id> backfill-kubernetes --help` lists every `--overwrite-*` and `--filter-*` flag. All filter timestamps (`--filter-contains`, `--filter-start`, `--filter-end`) must be full ISO with seconds precision, e.g. `2024-01-15T00:00:00`. Endpoint timestamps (`--append-dim-end`, `--filter-end`) are exclusive; `--filter-start` and `--filter-contains` are inclusive.

## Tuning parallelism

- **jobs_per_pod** — aim for jobs that take 3–15 minutes, to amortize pod startup and reduce icechunk commit compare-and-set contention. Materialized: 2–4 for non-ensemble datasets, 1 for ensemble.
- **max_parallelism** — materialized: 20–50. Much higher (~200) is often fine, but verify first that the cluster can fit that many of this dataset's pods: compare both the cpu and the memory one pod requests against available capacity, since either can be the binding constraint and the limit may be a quota rather than a node count. Leave headroom so operational updates can still schedule, and watch for unschedulable pods once the job starts. Some sources cap useful parallelism (`s3://ecmwf-forecasts` supports at most 8). Virtual: 6–10; higher risks compare-and-set contention because every worker commits to the same Icechunk branch.

### Sizing a virtual backfill

Commit latency, rebase contention and the parallelism ceiling are covered in [virtual_datasets.md](virtual_datasets.md#backfill-parallel-on-a-pre-sized-temp-branch) — commit latency is `(1 + rebase_attempts) × flush cost`, a flush rewrites every manifest window the session touches, every lost branch-tip race re-runs it, and the failure mode is a worker that re-flushes indefinitely once its flush outlasts its competitors' inter-commit spacing. Read that first; this section adds the one cost it does not cover and the order to size in.

**Two ref counts, two costs.** They are different quantities and must not be conflated:

- **Refs rewritten per commit** — arrays touched × refs per active manifest split — drives flush cost and therefore commit latency and rebase cost. It is set by `manifest_append_dim_split` and array count, not by `jobs_per_pod`.
- **Refs added per commit** — `jobs_per_pod × refs per job`, where a job is one chunk along the append dim and refs per job is every chunk reference that position carries (members × leads × spatial chunks, summed over arrays) — drives **memory**, because the worker holds what it is adding. Peak RSS runs about **1 GiB per million refs added**: measured 5.7M → 6.1 GiB and 13.2M → 13.5 GiB. A commit rewriting 12M refs on a 3.7G pod is routine when it adds only thousands; a commit adding 19M refs on a 15G pod lost most of its round to eviction.

Memory carries two further effects on top of that slope, both small: a rebasing commit re-merges and pays roughly 0.1–0.3 GiB more per concurrent sibling it rebases over, worst observed 41% above base; and the base itself moves during a run — one discontinuity of a few percent, plus slow drift of about the same total size over many hours, neither understood. Leave room for all three rather than fitting the request to the first round.

**The ceiling is the node, not the request.** Backfill pods set a memory request and no limit, so nothing is killed at the request; the failure is node-pressure eviction. The per-pod ceiling is `node allocatable / min(cpu_allocatable // cpu_request, mem_allocatable // mem_request)`, and `deploy/aws/nodepool.yaml` pins the instance classes, so the smallest class present is knowable before a run starts. Never express a monitoring threshold as a fixed number of gigabytes; compare a node's summed peaks against its own allocatable, across every job sharing it concurrently.

A backfill does not declare its own resources: it copies cpu and memory from the dataset's update `ReformatCronJob`, so changing them is a code change merged to main, and trimming a cron job's request shrinks every future backfill of that dataset.

**The order to size in:**

1. Take the memory request as given. Cap refs added per commit at about 60% of the request in GiB × 1M — a 7G request, 6.52 GiB, caps near 3.9M — which leaves room for the rebase and drift terms above.
2. Derive `jobs_per_pod` from that cap and the dataset's refs per job.
3. Check refs rewritten per commit against the split guidance in [virtual_datasets.md](virtual_datasets.md#manifest-splitting); if flush cost is the binding term, the fix is the split, not `jobs_per_pod`.
4. Set parallelism low and raise it while `rebase_attempts` stays small — read it from `repo.ancestry`, and watch the maximum, not the mean. Within one round of a 2,905-worker run, commit time rose from about 7s for the first commit to land to about 53s for the sixth, at constant refs: that is contention, and it scales with the number of concurrent writers, not the size of the commit. Parallelism raises in place with `kubectl patch`; the memory request does not.
5. Check pod duration last, and accept it outside the 3–15 minute band when the constraints above bind first.

`jobs_per_pod` does not transfer between datasets, because refs per job spans orders of magnitude — 38 for an analysis position carrying one per array, against ~1.9M for an ensemble forecast init carrying every lead × member. Compute it for the dataset in front of you.

For the cpu / memory / shared-memory a dataset's jobs request, see the Kubernetes resource values in [implementation_guide.md](implementation_guide.md) §5.

Parallelism beyond what the cluster can schedule starves it: operational update pods sit Pending, and worker 0 does setup before any other worker proceeds, so if worker 0 is not among the pods that scheduled, the workers that did start log `Waiting for worker 0 to complete setup...` and then exit. Indexed jobs retry those indices, so the backfill still finishes, but the attempts are wasted. To free capacity on a running job without losing progress, lower its parallelism in place — `kubectl patch job <name> -p '{"spec":{"parallelism":N}}'` — which does not evict running pods; capacity frees as they finish.

## Concurrency with operational updates

An operational update that publishes mid-backfill makes an overwrite backfill's finalize fail loudly (the update wins; re-run the backfill). Do **not** suspend an active update cron to avoid this — that delays the production pipeline. Instead run the backfill between update fires, splitting a long history into several smaller `filter_start`/`filter_end` backfills. See "Concurrent jobs writing to the same dataset" in [parallel_processing.md](parallel_processing.md).
