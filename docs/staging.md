# Staging datasets

Run multiple versions of a dataset's operational updates concurrently. Use cases:
- **Staging**: operationally test a new dataset version before promoting it
- **Legacy support**: keep an older version updating while users transition

## How it works

A naming convention links git branches, docker images, k8s cronjobs, and dataset versions:

| Component | Convention | Example |
|---|---|---|
| Git branch | `stage/{dataset_id}/v{version}` | `stage/noaa-gfs-forecast/v0.3.0` |
| K8s cronjobs | `staging-{dataset_id}-v{version}-{update\|validate}` | `staging-noaa-gfs-forecast-v0-3-0-update` |
| Store path | `{base_path}/{dataset_id}/v{version}.{ext}` | `s3://…/noaa-gfs-forecast/v0.3.0.zarr` |
| Docker image | `{repository}:{git_sha}` | Same convention as production |

Pushing to a `stage/**` branch triggers a GitHub Actions workflow that runs the full CI suite, builds a Docker image, and deploys cronjobs for only the named dataset.

## Setup a staged version

### 1. Prepare the code

On a feature branch, make your changes and bump the version in the dataset's `TemplateConfig.dataset_attributes`. The version must differ from what's on `main`. Run `uv run main <dataset-id> update-template` and commit the template changes.

### 2. Backfill

Before staging operational updates can run, the new version's store must exist. Run a backfill into the new version's store, the same process as any new dataset:

```bash
uv run main <dataset-id> backfill-kubernetes ...
```

Verify the backfill before proceeding.

### 3. Push to the staging branch

After CI passes on a PR to main:

```bash
git push origin my-feature:stage/noaa-gfs-forecast/v0.3.0
```

This triggers the `deploy-staging.yml` workflow which:
1. Runs ruff, ty, and the full pytest suite
2. Builds a Docker image
3. Validates the dataset ID exists, the version matches the template, and the version differs from main
4. Deploys staging cronjobs via `kubectl apply`

The staging cronjobs then run on their defined schedule, writing to the new version's store.

### 4. Iterate

Push updated code to the staging branch. Either merge or force push:

```bash
# Fast-forward merge
git checkout stage/noaa-gfs-forecast/v0.3.0
git merge another-feature
git push

# Or force push to replace
git push origin another-feature:stage/noaa-gfs-forecast/v0.3.0 --force
```

Each push reruns CI and redeploys the staging cronjobs with the new image.

## Promote to production

When the staged version is ready:

1. Merge the changes into `main` via a normal PR
2. Main's deploy workflow picks it up and deploys as the production version
3. Clean up the staging resources (see below)

## Clean up

When a staged version is no longer needed:

```bash
uv run main cleanup-staging --dataset-id noaa-gfs-forecast --version 0.3.0
```

This deletes:
- K8s cronjobs (`staging-noaa-gfs-forecast-v0-3-0-update`, `staging-noaa-gfs-forecast-v0-3-0-validate`)
- The remote git branch (`stage/noaa-gfs-forecast/v0.3.0`)

To also delete Sentry cron monitors, pass a [Sentry auth token](https://docs.sentry.io/account/auth-tokens/):

```bash
uv run main cleanup-staging --dataset-id noaa-gfs-forecast --version 0.3.0 --sentry-auth-token <token>
```

The dataset store is **not** deleted. Clean it up manually when ready (e.g. `aws s3 rm --recursive`).

## Constraints

- **One dataset per staging branch.** The branch name encodes a single dataset. To stage common code changes across multiple datasets, create separate staging branches from the same feature branch.
- **K8s name length.** Staging cronjob names must fit in 63 characters. This is validated at deploy time.
- **Source data load.** Multiple versions hitting upstream APIs on the same schedule increases load. Consider staggering the staging schedule on the staging branch.
- **Manual workflows.** The auto-generated manual GitHub workflows only list production cronjobs. Use `kubectl` directly for staging cronjobs.

## CLI reference

```
uv run main deploy-staging --dataset-id <id> --version <version> --docker-image <tag>
uv run main cleanup-staging --dataset-id <id> --version <version> [--sentry-auth-token <token>]
```
