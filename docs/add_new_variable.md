# Add new variable

How to add a new data variable to an existing dataset.

## 1. Add the variable to the template

1a. Download a real, recent source file for the new variable. Open it (e.g. with `rasterio`, `gdalinfo`, or `xarray`) to inspect its attributes (units, grid dimensions, CRS, nodata/sentinel values) and data values (range, NaN/missing coverage, geographic distribution). This grounds the variable configuration in observed data rather than assumptions.

1b. Add a new `DataVar` to your dataset’s `TemplateConfig.data_vars` (usually in `src/reformatters/<provider>/<model>/<variant>/template_config.py`).
   - **Name + externally visible attrs**: match existing naming/attrs used in this repo where possible; otherwise follow CF Conventions. Variable names generally follow the format `<long name>_<level>`.
   - **Internal attrs**: derive from `gdalinfo` output on a representative source file (and GRIB index if relevant)
   - **Encoding**: match existing variables, and follow the `keep_mantissa_bits` guidance in AGENTS.md.

1c. Regenerate the checked-in Zarr template metadata:

```bash
uv run main <DATASET_ID> update-template
```

1d. Edit the `test_backfill_local_and_operational_update` test in the dataset's dynamical_dataset_test.py to add the new variable. Run the test and add a quick print or plot to confirm data is successfully being processed for the new variable. Abandon these test changes after confirming to keep our integration tests from getting slow.

1e. Open and merge a PR containing the template changes (`template_config.py` and zarr metadata in `templates/latest.zarr/`)

## 2. Backfill data for the new variable

After the PR is merged to main, run a backfill filtered to just the new variable. The easiest way is the GitHub Action [Manual: Backfill](https://github.com/dynamical-org/reformatters/actions/workflows/manual-backfill.yml) (requires reformatters repo write access):

- **operation** = `overwrite-chunks-and-metadata` — refreshes the store's metadata from the template (creating the new variable) and writes its chunk data. The guards never trim the store, and its extent is unchanged unless you explicitly set an append_dim_end past the current end.
- **filter_variable_names** = your new variable's name.
- **jobs_per_pod** = 1 or 2 in most cases for materialized datasets; 30 for virtual. For both, the goal is jobs that take 3-15 minutes, to amortize startup time and reduce icechunk commit compare-and-set contention.
- **max_parallelism** = materialized: 100-300 if the data source supports highly parallel reads (100 is often sufficient; s3://ecmwf-forecasts supports at most 8). Virtual: 10 — any higher risks heavy compare-and-set contention.

Or the equivalent CLI (requires kubectl access to the cluster):

```bash
DYNAMICAL_ENV=prod uv run main <DATASET_ID> backfill-kubernetes \
  --overwrite-chunks --overwrite-metadata \
  --filter-variable-names <VARIABLE_NAME>
```

An operational update that publishes while the backfill runs makes the backfill's finalize fail loudly and the backfill must be re-run, so run the backfill in between update runs — for a long history, as several smaller `filter_start`/`filter_end` backfills (see "Concurrent jobs writing to the same dataset" in [docs/parallel_processing.md](parallel_processing.md)).

## 3. Validate

Follow [docs/validation.md](validation.md) — it walks through running `run-all`, reading `validation_summary.md`, inspecting every plot, and the full data quality checklist. When validating a new variable it is often useful to restrict with `--variable <name>` to iterate faster.

## 4. Update dataset catalog documentation

Update the dataset catalog docs on `dynamical.org` by rebuilding (`npm run build`) and merging updates to main in `https://github.com/dynamical-org/dynamical.org`.
