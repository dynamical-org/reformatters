# Staging datasets

Run multiple versions of a dataset's operational updates concurrently. Use cases:
- **Staging**: operationally test a new dataset version before promoting it
- **Legacy support**: keep an older version updating while users transition

## How it works

A naming convention links git branches, kubernetes cronjobs, and dataset versions:

| Component | Convention | Example |
|---|---|---|
| Git branch | `stage/{dataset_id}/v{version}` | `stage/noaa-gfs-forecast/v0.3.0` |
| Kubernetes cronjobs | `stage-{dataset_id}-v{version}-{update\|validate}` | `stage-noaa-gfs-forecast-v0-3-0-update` |
| Store path | `{base_path}/{dataset_id}/v{version}.{ext}` | `s3://…/noaa-gfs-forecast/v0.3.0.zarr` |

Pushing to a `stage/**` branch triggers a GitHub Actions workflow that runs the full CI suite, builds a Docker image, and deploys cronjobs (e.g. update and validate) for only the specific dataset in the branch name.

## Setup a staged version

### 1. Prepare the code

On a feature branch, make your changes and bump the version in the dataset's `TemplateConfig.dataset_attributes`. The version must differ from what's on `main`. Run `uv run main <dataset-id> update-template` and commit the template changes.

### 2. Backfill

Before staging operational updates can run, the new version's store must exist. From your feature branch, run a backfill into the new version's store, the same process as any new dataset:

```bash
uv run main <dataset-id> backfill-kubernetes ...
```

### 3. Push to the staging branch

Once you're ready to deploy to staging, push your feature branch to a staging branch:

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

Preview what will be deleted:

```bash
uv run main cleanup-staging noaa-gfs-forecast 0.3.0
```

Then run with `--force` to execute:

```bash
uv run main cleanup-staging noaa-gfs-forecast 0.3.0 --force
```

This deletes:
- Kubernetes cronjobs (`stage-noaa-gfs-forecast-v0-3-0-update`, `stage-noaa-gfs-forecast-v0-3-0-validate`)
- The remote git branch (`stage/noaa-gfs-forecast/v0.3.0`)

The dataset store and Sentry cron monitors are **not** deleted. Clean them up manually when ready.

## Constraints

- **One dataset per staging branch.** The branch name encodes a single dataset. To stage common code changes across multiple datasets, create separate staging branches from the same feature branch.
- **Kubernetes name length.** Staging cronjob names must fit in 63 characters. Long dataset IDs are automatically trimmed to fit.
- **Manual workflows.** The auto-generated manual GitHub Create Job from CronJob workflows only list production cronjobs. Use `kubectl` directly for staging cronjobs.

## CLI reference

```
uv run main deploy-staging <dataset-id> <version> <docker-image>
uv run main cleanup-staging <dataset-id> <version> [--force]
```
