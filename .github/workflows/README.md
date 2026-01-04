# GitHub Actions Workflows

## Manual Kubernetes Operations Workflows

These workflows allow team members to manually trigger specific Kubernetes operations from GitHub.

### Available Workflows

#### 1. Create Job from CronJob
**File:** `manual-create-job-from-cronjob.yml`

Manually trigger a one-off job from an existing cronjob.

- **Input:** Dropdown list of all available cronjobs (auto-updated via pre-commit)
- **Job naming:** Auto-generates job names in format: `{cronjobname}-{username}-{random}`
  - Automatically truncates to meet Kubernetes 63-character limit
- **Output:** Job summary displays created job details with link to check status
- **Usage:** 
  1. Go to Actions tab → "Manual: Create Job from CronJob"
  2. Click "Run workflow"
  3. Select cronjob from dropdown
  4. Click "Run workflow" button

#### 2. Get Jobs
**File:** `manual-get-jobs.yml`

List all jobs with their status, sorted by creation time.

- **Output:** Job summary displays formatted table of all jobs at the top of the workflow run page
- **Logs:** Full kubectl output is also available in step logs

#### 3. Get Pods
**File:** `manual-get-pods.yml`

List all pods with their status, sorted by creation time.

- **Output:** Job summary displays formatted table of all pods at the top of the workflow run page
- **Logs:** Full kubectl output is also available in step logs

### Job Summaries

All manual workflows now write useful output to **Job Summaries**, which appear prominently at the top of each workflow run page (above the step logs). This makes it much easier to find the important information without digging through logs.

- `Get Jobs` and `Get Pods` display the kubectl output in a formatted code block
- `Create Job from CronJob` displays the created job details with next steps

To see a job summary, look for the summary section at the top of the workflow run page after clicking into a workflow run.

### Access Control

Anyone with write access to this repository can trigger these workflows.

### How the Dropdown Auto-Updates

The cronjob dropdown in `manual-create-job-from-cronjob.yml` is automatically generated:

1. **Generator Script:** `src/scripts/generate_manual_workflows.py`
   - Scans `DYNAMICAL_DATASETS` in `src/reformatters/__main__.py`
   - Extracts all `CronJob` instances from `operational_kubernetes_resources()`
   - Generates workflow YAML with dropdown choices

2. **Pre-commit Hook:** Configured in `.pre-commit-config.yaml`
   - Automatically runs generator when relevant files change
   - Triggers on changes to:
     - `src/reformatters/__main__.py`
     - `src/reformatters/*/dynamical_dataset.py`
     - `src/scripts/generate_manual_workflows.py`

### Manual Regeneration

If needed, you can manually regenerate the workflow files:

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
