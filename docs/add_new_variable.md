# Add new variable

How to add a new data variable to an existing dataset.

## 1. Add the variable to the template

1a. Add a new `DataVar` to your dataset’s `TemplateConfig.data_vars` (usually in `src/reformatters/<provider>/<model>/<variant>/template_config.py`).
   - **Name + externally visible attrs**: match existing naming/attrs used in this repo where possible; otherwise follow CF Conventions. Variable names generally follow the format `<long name>_<level>`.
   - **Internal attrs**: derive from `gdalinfo` output on a representative source file (and GRIB index if relevant)
   - **Encoding**: match existing variables, setting `keep_mantissa_bits` to 7 by default, 6 for wind variables, and 10 for pressure variables with units `pa`.

1b. Regenerate the checked-in Zarr template metadata:

```bash
uv run main <DATASET_ID> update-template
```

1c. Edit the `test_backfill_local_and_operational_update` test in the dataset's dynamical_dataset_test.py to add the new variable. Run the test and add a quick print or plot to confirm data is successfully being processed for the new variable. Abandon these test changes after confirming to keep our integration tests from getting slow.

1d. Open and merge a PR containing the template changes (`template_config.py` and zarr metadata in `templates/latest.zarr/`)

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

Run the plotting tools and inspect the generated images in `data/output/<dataset-id>/`.

```bash
uv run python src/scripts/validation/plots.py run-all <DATASET_URL>
```

Common issues to look out for:
- Unexpected missing data (nulls/holes where you expect coverage).
- Units vs values mismatch (e.g. mm/h vs mm/s).
- Over-quantization (step changes in spatial or time series plots due to a too low `keep_mantissa_bits` value)
- Time misalignment (e.g. diurnal cycle peaks shifted vs a reference dataset).

Notes
- `DATASET_URL` is the complete, direct URL to the dataset (`bucket-prefix/dataset-id/version`), e.g. `s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.zarr`. The bucket prefix can be found in `__main__.py` and the dataset id and version in the `TemplateConfig.dataset_attributes`.
- The spatial and timeseries plots will plot the data against a reference dataset (GEFS analysis by default) to highlight unexpected differences.
- You can also run each validation plot individually, see `uv run src/scripts/validation/plots.py --help`.
- You can add additional `--variable` flags if side by side plots help add context (e.g. show solar radiation alongside cloud cover).

## 4. Update dataset catalog documentation

Update the dataset catalog docs on `dynamical.org` by rebuilding (`npm run build`) and merging updates to main in `https://github.com/dynamical-org/dynamical.org`.