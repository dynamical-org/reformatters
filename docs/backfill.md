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

For virtual backfills, size `jobs_per_pod` by refs per commit (`jobs_per_pod × refs per job`); roughly 0.3–0.5 million refs has kept commits under about 10 seconds in measured datasets. Commit latency grows faster than linearly above that range: a GFS forecast backfill at 4.4 million refs per commit measured p50 26 seconds, p95 88 seconds, and maximum 418 seconds. Across datasets, array count matters because each commit read-modify-writes one manifest per array it touches: GFS analysis (~298 arrays, 0.52 million refs per commit) measured p50 7.8 seconds, while GEFS 10 day (38 arrays, 0.34 million refs per commit) measured median 1.7 seconds and maximum 3.1 seconds. A dataset with few arrays can therefore exceed the range, while one with many arrays should stay at or below it.

Do not reuse `jobs_per_pod` across datasets; refs per job drive the choice: sizing for two products from the same model family and source calls for 16 for a forecast with 95,418 refs per region job (lead times × ensemble members) and 3,000 for an analysis with 38 refs per job (one per array). Run a small filtered backfill first, then use its pods' durations and commit times to select the value. When refs per commit is not binding, allow about two minutes of actual work on top of the roughly 65-second pod startup floor measured across different virtual backfills; with less than two minutes of work, startup takes most of the pod's lifetime. At 18.5 seconds per region job, for example, 4 jobs make roughly 74-second, startup-dominated pods, while 16 make roughly 5-minute pods and cut pod count fourfold.

For the cpu / memory / shared-memory a dataset's jobs request, see the Kubernetes resource values in [implementation_guide.md](implementation_guide.md) §5.

Parallelism beyond what the cluster can schedule starves it: operational update pods sit Pending, and worker 0 does setup before any other worker proceeds, so if worker 0 is not among the pods that scheduled, the workers that did start log `Waiting for worker 0 to complete setup...` and then exit. Indexed jobs retry those indices, so the backfill still finishes, but the attempts are wasted. To free capacity on a running job without losing progress, lower its parallelism in place — `kubectl patch job <name> -p '{"spec":{"parallelism":N}}'` — which does not evict running pods; capacity frees as they finish.

## Concurrency with operational updates

An operational update that publishes mid-backfill makes an overwrite backfill's finalize fail loudly (the update wins; re-run the backfill). Do **not** suspend an active update cron to avoid this — that delays the production pipeline. Instead run the backfill between update fires, splitting a long history into several smaller `filter_start`/`filter_end` backfills. See "Concurrent jobs writing to the same dataset" in [parallel_processing.md](parallel_processing.md).
