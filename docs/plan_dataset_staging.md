# Plan: Dataset staging / parallel deploys

## Goal

Run operational updates of multiple versions of a dataset concurrently. This enables:
- **Staging**: operationally test a new dataset version before promoting it to primary
- **Legacy support**: keep one or more legacy versions updating while users transition to a new version

## Linking convention

A shared naming convention links git branches, docker images, k8s cronjobs, and dataset versions:

| Component | Convention | Example |
|---|---|---|
| Git branch | `stage/{dataset_id}/v{version}` | `stage/noaa-gfs-forecast/v0.3.0` |
| Docker image | `{repository}:{git_sha}` (same as main) | `ecr.../reformatters:abc123` |
| K8s cronjobs | `staging-{dataset_id}-v{version_dashes}-{update\|validate}` | `staging-noaa-gfs-forecast-v0-3-0-update` |
| Store path | `{base_path}/{dataset_id}/v{version}.{ext}` (existing convention) | `s3://…/noaa-gfs-forecast/v0.3.0.zarr` |

Dots in the version are replaced with dashes in k8s resource names (DNS name constraint).

## Workflow

### Setup (manual, one-time per staged version)

1. Create a feature branch with the dataset changes (new version in template config, code changes, etc.)
2. Open a PR against main. CI runs code quality checks.
3. Once CI passes and review is complete, create the staging branch: `stage/{dataset_id}/v{version}`
4. Merge the feature branch into the staging branch.
5. Run a backfill into the new version's store (same manual process as for any new dataset version).
6. Verify the backfill.

### Operational updates (automated)

Pushing to a `stage/**` branch triggers a GitHub Actions workflow that:

1. Runs the full code quality suite (ruff check, ruff format, ty check, pytest)
2. Parses `dataset_id` and `version` from the branch name
3. Builds and pushes a Docker image (same multi-arch Depot build as main)
4. Runs `uv run main deploy-staging --dataset-id {id} --version {version} --docker-image {tag}`

The `deploy-staging` command validates and deploys (details below).

### Iterating on staging

To update the staged version's code: merge new feature branches into the staging branch. Each push triggers the workflow above, rebuilding the image and updating the cronjobs.

### Promoting to primary

When the staged version is ready to become the primary version:

1. Merge the staging branch's changes into main via a PR
2. Main's deploy workflow picks it up, deploying the version as the primary
3. Clean up the now-redundant staging resources (see cleanup below)

## Components to build

### 1. GitHub Actions workflow: `deploy-staging.yml`

Triggers on push to `stage/**` branches.

```yaml
on:
  push:
    branches: [ 'stage/**' ]

concurrency:
  # Per-branch concurrency so staging deploys don't block each other or main
  group: "deploy-staging-${{ github.ref_name }}"
  cancel-in-progress: false
```

Steps:
1. Checkout, install uv, python, project (same as code-quality.yml)
2. Run `uv run ruff check`, `uv run ruff format --check`, `uv run ty check`, `uv run pytest` — full suite, enforced before deploy
3. Parse dataset_id and version from `${{ github.ref_name }}` (strip `stage/` prefix, split on `/v`)
4. Configure AWS, kubectl, Depot (same as deploy-operational-updates.yml)
5. Build and push Docker image
6. Run `uv run main deploy-staging --dataset-id $DATASET_ID --version $VERSION --docker-image $IMAGE_TAG`

### 2. CLI command: `deploy-staging`

New command in `__main__.py`:

```
uv run main deploy-staging --dataset-id {id} --version {version} --docker-image {tag}
```

#### Validations

1. **Dataset ID is valid**: `dataset_id` must match a dataset in `DYNAMICAL_DATASETS`
2. **Version matches template config**: the dataset's `template_config.version` must equal the `--version` argument
3. **Version differs from main**: use `git show main:{template_zarr_path}/zarr.json` to read main's `dataset_version` attribute and confirm it's different. This prevents a staging deploy from creating cronjobs that conflict with production.

#### Deploy logic

Reuses existing infrastructure with minimal changes:

1. Find the matching dataset in `DYNAMICAL_DATASETS`
2. Call `dataset.operational_kubernetes_resources(image_tag)` to get the cronjob definitions
3. Use `replace()` from `common/pydantic.py` to create copies with staging-prefixed names:
   - `{dataset_id}-update` → `staging-{dataset_id}-v{ver_dashes}-update`
   - `{dataset_id}-validate` → `staging-{dataset_id}-v{ver_dashes}-validate`
4. Apply to k8s via `kubectl apply -f -` (same as `deploy_operational_resources`)

#### Changes to `deploy.py`

Add an optional `dataset_id_filter` parameter to `deploy_operational_resources()`:

```python
def deploy_operational_resources(
    datasets: Iterable[DynamicalDataset[Any, Any]],
    docker_image: str | None = None,
    dataset_id_filter: str | None = None,
    cronjob_name_fn: Callable[[str], str] | None = None,
) -> None:
```

When `dataset_id_filter` is set, only deploy that dataset's cronjobs. When `cronjob_name_fn` is provided, use it to transform cronjob names (via `replace(cronjob, name=cronjob_name_fn(cronjob.name))`). The existing `deploy` command and main workflow pass neither argument and behave exactly as today.

The `deploy-staging` command calls this with both arguments set.

### 3. CLI command: `cleanup-staging`

```
uv run main cleanup-staging --dataset-id {id} --version {version}
```

Run locally (requires kubectl and git access). Steps:

1. **Validate** dataset_id exists in `DYNAMICAL_DATASETS`
2. **Delete k8s cronjobs**:
   ```
   kubectl delete cronjob staging-{dataset_id}-v{ver_dashes}-update staging-{dataset_id}-v{ver_dashes}-validate
   ```
3. **Delete Sentry cron monitors** via Sentry API (using monitor slugs matching the cronjob names). Requires a Sentry API token (could be passed as `--sentry-token` or read from env).
4. **Delete the git branch**:
   ```
   git push origin --delete stage/{dataset_id}/v{version}
   ```
5. **Print what was left behind** for manual cleanup:
   ```
   Not deleted (clean up manually when ready):
     s3://…/noaa-gfs-forecast/v0.3.0.zarr
   ```

No version-vs-main check here. A common flow is: merge staging changes to main (promoting the version), then clean up the now-duplicate staging resources.

### 4. Sentry monitoring

Staging cronjobs automatically create Sentry cron monitors because the existing `_monitor()` context manager in `DynamicalDataset` uses the cronjob name as the monitor slug, and the `CRON_JOB_NAME` env var is set from the cronjob's `.name` field. Since we rename the cronjobs with the `staging-` prefix, the Sentry monitors will have distinct slugs. No code changes needed for monitoring to work — `cleanup-staging` handles deleting the monitors.

## Design constraints and notes

### One dataset per staging branch

Each staging branch is scoped to a single dataset. The branch name encodes one `dataset_id`. To stage changes to common code affecting multiple datasets, create multiple staging branches (one per dataset) from the same feature branch.

### Store isolation

Store paths include the version (`v{version}.{extension}`), so a staging version writes to a completely separate location from production. No risk of data interference by construction.

### Source data server load

Multiple versions of the same dataset hitting upstream APIs (NOAA, ECMWF, etc.) on the same schedule could increase load. Consider staggering the staging cronjob schedule by a few minutes. This could be done manually by adjusting the schedule in the dataset's `operational_kubernetes_resources` on the staging branch, or automatically by the staging deploy adding a small offset.

### K8s name length

K8s resource names are limited to 63 characters. The staging prefix plus version consumes ~15 characters. The longest current dataset ID is `ecmwf-ifs-ens-forecast-15-day-0-25-degree` (42 chars). With staging: `staging-ecmwf-ifs-ens-forecast-15-day-0-25-degree-v0-3-0-update` = 63 chars. This is tight but fits. Worth adding a validation check in `deploy-staging` that asserts the generated name is <= 63 chars.

### Manual workflow dropdowns

The auto-generated manual workflows (`generate_manual_workflows.py`) list cronjobs from `DYNAMICAL_DATASETS` on main. Staging cronjobs won't appear there. Use `kubectl` directly to interact with staging cronjobs. This is acceptable since staging is a temporary, operator-managed state.

### Docker image scope

The Docker image built from a staging branch contains code for ALL datasets, not just the staged one. This is fine — staging cronjobs only invoke the staged dataset's commands. But operators should avoid using a staging image for other datasets' manual jobs.

### Branch synchronization

If shared code in `common/` changes on main, the staging branch should pick up those changes via periodic merges from main. Conflicts are unlikely since the staging branch primarily modifies one dataset's files.

### Race conditions between main and staging deploys

Main deploys ALL cronjobs. Staging deploys only the staged dataset's cronjobs. Since the cronjob names are different (production vs staging-prefixed), `kubectl apply` won't interfere. However, if main deploys at the same time as a staging deploy, there's no conflict because they're applying different k8s resources. The separate concurrency groups in GitHub Actions handle workflow-level serialization.

### What main's deploy does NOT do

Main's existing deploy workflow applies production cronjobs without staging awareness. It does not delete staging cronjobs, and staging cronjobs (with different names) are unaffected. The only scenario requiring care is: if main is updated to the same version as an active staging branch, both production and staging cronjobs would write to the same store path. The version-differs-from-main check in `deploy-staging` prevents creating this situation, but doesn't prevent main from later bumping to match an existing staging version. This is an acceptable operational risk — the cleanup step should happen promptly after promoting a version.

## Files to create or modify

| File | Change |
|---|---|
| `.github/workflows/deploy-staging.yml` | New workflow |
| `src/reformatters/__main__.py` | Add `deploy-staging` and `cleanup-staging` commands |
| `src/reformatters/common/deploy.py` | Add `dataset_id_filter` and `cronjob_name_fn` params to `deploy_operational_resources` |
| `src/reformatters/common/staging.py` | New module: staging validation logic (parse branch name, version checks, cronjob name generation) |
| `tests/common/staging_test.py` | Tests for staging validation and name generation |
