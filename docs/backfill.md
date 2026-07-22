# Backfill

Populate a dataset's store — create and fill a brand-new store, add a newly-implemented variable, or re-run the positions a validation pass flagged. Backfills distribute work across Kubernetes indexed jobs the same way operational updates do; see [parallel_processing.md](parallel_processing.md) for how workers coordinate.

Run a backfill only after the dataset's code is merged to `main`: the GitHub action and the deployed container image both build from `main`, so this keeps the driver and workers on the same commit.

## Prerequisites (new dataset)

- Create the public bucket once: `./deploy/aws/create_new_aws_open_data_bucket.sh <provider>-<model>`.
- To enable Sentry issue reporting and cron monitoring, create the secret once per cluster: `kubectl create secret generic sentry --from-literal='DYNAMICAL_SENTRY_DSN=xxx'`.
- Contact feedback@dynamical.org if you are setting up new compute or storage.

## Where to run

- **GitHub Action (preferred).** The [Manual: Backfill](https://github.com/dynamical-org/reformatters/actions/workflows/manual-backfill.yml) action (workflow_dispatch, requires repo write access) runs only from `main`, waits for main's tip to finish deploying, and submits the job with that deploy's image. It exposes only the safe operations.
- **Kubernetes from your machine.** `DYNAMICAL_ENV=prod uv run main <dataset-id> backfill-kubernetes [flags]`, then track with `kubectl get jobs`. Complete README.md > Deploying to the cloud > Setup first.
- **Local.** `DYNAMICAL_ENV=prod uv run main <dataset-id> backfill-local <append-dim-end>`. Simple and fine when the dataset fits your disk and time budget.

## Operations

Pick the operation by what you're doing (action operation name / equivalent CLI flags):

- **New store** — `create-new-store` / `backfill-kubernetes`. Creates the store and fails if one already exists. `append_dim_end` defaults to now (leave it empty to backfill through now).
- **New variable, or refresh metadata** — `overwrite-chunks-and-metadata` / `backfill-kubernetes --overwrite-chunks --overwrite-metadata --filter-variable-names <name>`. Refreshes metadata from the template (creating newly-added variables; the guards never trim the store) and writes the filtered chunk data. The store extent is unchanged unless you set an `append_dim_end` past the current end.
- **Re-backfill flagged positions** — the same overwrite operation, filtered to the variables and the `filter_start`/`filter_end` window a validation pass flagged.

`uv run main <dataset-id> backfill-kubernetes --help` lists every `--overwrite-*` and `--filter-*` flag.

## Tuning parallelism

- **jobs_per_pod** — aim for jobs that take 3–15 minutes, to amortize pod startup and reduce icechunk commit compare-and-set contention. Materialized: 1–2 in most cases. Virtual: ~30.
- **max_parallelism** — materialized: 100–300 if the source supports highly parallel reads (100 is often enough; `s3://ecmwf-forecasts` supports at most 8). Virtual: 10 — higher risks heavy compare-and-set contention.

For the cpu / memory / shared-memory a dataset's jobs request, see the Kubernetes resource values in [implementation_guide.md](implementation_guide.md) §5.

## Concurrency with operational updates

An operational update that publishes mid-backfill makes an overwrite backfill's finalize fail loudly (the update wins; re-run the backfill). Do **not** suspend an active update cron to avoid this — that delays the production pipeline. Instead run the backfill between update fires, splitting a long history into several smaller `filter_start`/`filter_end` backfills. See "Concurrent jobs writing to the same dataset" in [parallel_processing.md](parallel_processing.md).
