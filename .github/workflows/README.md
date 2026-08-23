# GitHub Actions Workflows

## Manual Kubernetes Operations Workflows

These workflows allow team members to manually trigger specific Kubernetes operations from GitHub.

### Available Workflows

#### 1. Create Job from CronJob
**File:** `manual-create-job-from-cronjob.yml`

Manually trigger a one-off job from an existing cronjob.

- **Input:** Dropdown list of all available cronjobs (auto-updated via prek hook)
- **Job naming:** Auto-generates job names in format: `{cronjobname}-{username}-{random}`
  - Automatically truncates to meet Kubernetes 63-character limit
- **Usage:** 
  1. Go to Actions tab → "Manual: Create Job from CronJob"
  2. Click "Run workflow"
  3. Select cronjob from dropdown
  4. Click "Run workflow" button

#### 2. Get Jobs
**File:** `manual-get-jobs.yml`

List all jobs with their status, sorted by creation time.

#### 3. Get Pods
**File:** `manual-get-pods.yml`

List all pods with their status, sorted by creation time.

### Access Control

Anyone with write access to this repository can trigger these workflows.

### How the Dropdowns Auto-Update

The cronjob dropdown in `manual-create-job-from-cronjob.yml` and dataset dropdown in
`manual-backfill.yml` are automatically updated:

1. **Generator Script:** `src/scripts/generate_manual_workflows.py`
   - Scans `DYNAMICAL_DATASETS` in `src/reformatters/__main__.py`
   - Computes the available cronjobs and backfillable datasets
   - Updates only the dropdown choices in the checked-in workflows
   - Leaves the rest of each workflow unchanged

2. **Prek Hook:** Configured in `.pre-commit-config.yaml`
   - Automatically runs generator when relevant files change
   - Triggers on changes to:
     - `src/reformatters/__main__.py`
     - `src/reformatters/*/dynamical_dataset.py`
     - `src/scripts/generate_manual_workflows.py`

### Manual Regeneration

If needed, you can manually update the workflow choices:

```bash
uv run src/scripts/generate_manual_workflows.py
```

## Security

To limit our vulnerability to supply chain attacks on GitHub Actions we:

- Restrict Actions to those that we've written, or those published by GitHub or another verified developer
- [Pin versions](https://michaelheap.com/pin-your-github-actions/) to avoid unintentionally pulling in malicious code

### Pinning Versions

Run `npx pin-github-action .github/workflows` from the root of the repo to resolve the SHA for the given version tag of each Action.
This will ensure that we always use the exact version we expect, rather than whatever the latest commit for that tag is.

### Updating Pinned Versions

Dependabot should let us know when the Actions we're using have a new version available.

You can also run `npx pin-github-action .github/workflows` (from the root of the repo) to update the pinned SHAs for the given version tag of each action.
