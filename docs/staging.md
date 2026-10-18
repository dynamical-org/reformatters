# Staging datasets

Use `stage/{dataset_id}/v{version}` branches to run one dataset version alongside production. The version must match the template and differ from `main`; see [staging.py](../src/reformatters/common/staging.py) for naming and validation.

1. Prepare the code, bump the template version, and regenerate the template. Keep a staging-only version bump off the feature branch when testing a change that needs no production version bump.
2. [Backfill](backfill.md) the new version's store before enabling its updates.
3. Push to the staging branch. Each push runs CI, builds an image, and deploys that dataset's staging CronJobs through [deploy-staging.yml](../.github/workflows/deploy-staging.yml).
4. After testing, merge the production changes through a PR and clean up staging.

Use a separate staging branch for each dataset. Generated manual workflows list production CronJobs; use `kubectl` for staging jobs.

```bash
uv run main deploy-staging <dataset-id> <version> <docker-image>
uv run main cleanup-staging <dataset-id> <version>
uv run main cleanup-staging <dataset-id> <version> --force
```

Cleanup previews by default; `--force` deletes the staging CronJobs and remote branch. It leaves the store and Sentry monitors for manual cleanup. The next deploy removes deleted CronJobs from the trigger Role; admission bindings remain and deny missing templates.
