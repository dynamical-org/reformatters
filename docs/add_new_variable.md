# Add new variable

How to add a new data variable to an existing dataset. This is the Implement stage of the add-a-variable branch of the [dataset development guide](dataset_development_guide.md); backfill, validate, and publish are the shared stages there.

## Add the variable to the template

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

## Next

Once the PR is merged to `main`, backfill the new variable (an `overwrite-chunks-and-metadata` backfill filtered to it — see [backfill.md](backfill.md)), then validate and publish — the remaining stages of the [dataset development guide](dataset_development_guide.md).
