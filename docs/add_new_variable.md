# Add new variable

How to add a new data variable to an existing dataset.

## 1. Add the variable to the template

1a. Download a real, recent source file for the new variable. Open it (e.g. with `rasterio`, `gdalinfo`, or `xarray`) to inspect its attributes (units, grid dimensions, CRS, nodata/sentinel values) and data values (range, NaN/missing coverage, geographic distribution). This grounds the variable configuration in observed data rather than assumptions.

1b. Add a new `DataVar` to your dataset’s `TemplateConfig.data_vars` (usually in `src/reformatters/<provider>/<model>/<variant>/template_config.py`).
   - **Name + externally visible attrs**: match existing naming/attrs used in this repo where possible; otherwise follow CF Conventions. Variable names generally follow the format `<long name>_<level>`.
   - **Internal attrs**: derive from `gdalinfo` output on a representative source file (and GRIB index if relevant)
   - **Encoding**: match existing variables. Set `keep_mantissa_bits` per the defaults-by-variable-kind table in the [Encoding conventions](../AGENTS.md#encoding-conventions) section of AGENTS.md.

1c. Regenerate the checked-in Zarr template metadata:

```bash
uv run main <DATASET_ID> update-template
```

1d. Edit the `test_backfill_local_and_operational_update` test in the dataset's dynamical_dataset_test.py to add the new variable. Run the test and add a quick print or plot to confirm data is successfully being processed for the new variable. Abandon these test changes after confirming to keep our integration tests from getting slow.

1e. Open and merge a PR containing the template changes (`template_config.py` and zarr metadata in `templates/latest.zarr/`)

## 2. Write data for the new variable

#### 2a. Run an operational update

After the PR is merged, run an operational update once (or wait for the next scheduled run) so the dataset’s Zarr metadata includes the new variable before you backfill history.

To run it immediately, use the GitHub Action [Manual: Create Job from CronJob](https://github.com/dynamical-org/reformatters/actions/workflows/manual-create-job-from-cronjob.yml) (requires reformatters repo write access) setting "CronJob to create job from" to `{dataset-id}-update`.

#### 2b. Backfill history for the variable

> Prerequisite: You can build and push Docker images (see `README.md` > Deploying to the cloud > Setup).
> - `DOCKER_REPOSITORY` is set in your shell.
> - Your local Docker is authenticated to that repo.

Run a backfill filtered to just the new variable:

```bash
DYNAMICAL_ENV=prod uv run main <DATASET_ID> backfill-kubernetes \
  <APPEND_DIM_END> <JOBS_PER_POD> <MAX_PARALLELISM> \
  --overwrite-existing \
  --filter-variable-names <VARIABLE_NAME>
```

- `APPEND_DIM_END` = the current approximate UTC timestamp. The operational update in 2a will have already filled the latest data, so the exact value just needs to be somewhere around the present time.
- `JOBS_PER_POD` = 1 or 2 in most cases. Set to 3-4 if the jobs run very fast to amortize container startup time.
- `MAX_PARALLELISM` = 100 if the data source can support highly parallel reads, else lower.
- `--overwrite-existing` writes into the existing store rather than attempting (and failing) to create a new one.
- `--filter-variable-names` = your new variable's name.

## 3. Validate

Follow [docs/validation.md](validation.md) — it walks through running `run-all`, reading `validation_summary.md`, inspecting every plot, and the full data quality checklist. When validating a new variable it is often useful to restrict with `--variable <name>` to iterate faster.

## 4. Update dataset catalog documentation

Update the dataset catalog docs on `dynamical.org` by rebuilding (`npm run build`) and merging updates to main in `https://github.com/dynamical-org/dynamical.org`.
