# Add a new variable

Add a new data variable to an **existing** dataset (i.e., the Zarr store already exists).

## Prereqs

- You can build and push Docker images (see `README.md` > Deploying to the cloud > Setup).
  - `DOCKER_REPOSITORY` is set in your shell.
  - Your local Docker is authenticated to that repo.
- You have (at least) one representative source file handy for inspecting metadata (e.g., via `gdalinfo`), plus any GRIB index files if applicable.

## 1. Add the variable to the template

1. Add a new `DataVar` to your dataset’s `TemplateConfig.data_vars` (usually in `src/reformatters/<provider>/<model>/<variant>/template_config.py`).
   - **Name + externally visible attrs**: match existing naming/attrs used in this repo where possible; otherwise follow CF Conventions.
   - **Internal attrs / encoding hints**: derive from `gdalinfo` output on a representative source file and (if relevant) GRIB index metadata.

1. Regenerate the checked-in Zarr template metadata:

```bash
uv run main <DATASET_ID> update-template
git add src/reformatters/**/templates/latest.zarr
```

1. Open and merge a PR containing the template change.

## 2. Write data for the new variable

### 2a. Ensure an operational update runs (metadata updated last)

After the PR is merged, run an operational update once (or wait for the next scheduled run) so the dataset’s Zarr metadata includes the new variable before you backfill history.

To run it immediately, use the GitHub Action “Manual: Create Job from CronJob” (see `.github/workflows/manual-create-job-from-cronjob.yml`) to create a job from the dataset’s **update** CronJob.

### 2b. Backfill history for the variable (Kubernetes indexed job)

Run a backfill filtered to just the new variable:

```bash
DYNAMICAL_ENV=prod uv run main <DATASET_ID> backfill-kubernetes <APPEND_DIM_END> <JOBS_PER_POD> <MAX_PARALLELISM> \
  --overwrite-existing \
  --filter-variable-names <VARIABLE_NAME>
```

Notes:
- `--overwrite-existing` skips writing metadata at the start of the backfill (expected when writing into an existing store).
- If you need to backfill only a subset of time, the CLI also supports `--filter-start`, `--filter-end`, and `--filter-contains`.

## 3. Validate

Run the plotting tools and inspect the generated images in `data/output/`.

Common checks:
- **Unexpected missing data** (nulls/holes where you expect coverage).
- **Units vs values mismatch** (e.g., Kelvin-looking values labeled as °C).
- **Over-quantization** (blocky artifacts / heavy rounding due to encoding choices).
- **Time misalignment** (e.g., diurnal cycle peaks shifted vs a reference dataset).

### Null reporting

```bash
uv run python -m scripts.validation.plots report-nulls <DATASET_ZARR_URL> --variable <VARIABLE_NAME>
```

### Spatial comparison (vs a reference dataset)

```bash
uv run python -m scripts.validation.plots compare-spatial <DATASET_ZARR_URL> --variable <VARIABLE_NAME> --reference-url <REFERENCE_ZARR_URL>
```

### Timeseries comparison (vs a reference dataset)

```bash
uv run python -m scripts.validation.plots compare-timeseries <DATASET_ZARR_URL> --variable <VARIABLE_NAME> --reference-url <REFERENCE_ZARR_URL>
```

## 4. Update dataset catalog documentation

Update the dataset catalog docs in `dynamical.org` and deploy:
- Repo: `https://github.com/dynamical-org/dynamical.org`

## Open questions / clarifications

If I were following this end-to-end, I’d need answers to:

- **Which CronJob name(s)** correspond to “update” for a given `<DATASET_ID>` (and where to look them up)?
- **What is the canonical `<DATASET_ZARR_URL>` format** to use for validation (e.g., `https://data.dynamical.org/<dataset-path>/latest.zarr`)?
- **What should `<APPEND_DIM_END>` be** when adding a variable: “current dataset end date”, “now”, or “latest available in source data”?
- **Recommended backfill parameters**: what values do we usually use for `<JOBS_PER_POD>` and `<MAX_PARALLELISM>`?
- **Reference datasets**: what should we use as the default/reference for spatial and timeseries comparisons per dataset type (analysis vs forecast)?
- **Validation scope**: should plots default to a particular time window (e.g., last 7–30 days) to keep runs fast/cheap?
- **Acceptance thresholds**: what counts as “too many” nulls, “too much” quantization, or “too much” timing drift before we block release?
- **Docs deployment details**: what’s the standard process/commands for rebuilding and deploying in `dynamical.org` (and where should we link to it)?
