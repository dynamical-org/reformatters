# Add a new variable

Follow the shared stages in the [dataset development guide](dataset_development_guide.md), with these implementation steps:

1. Inspect a real source file: quantity, units, grid, CRS, missing values, and spatial coverage.
2. Add the `DataVar` to `TemplateConfig.data_vars`. Match equivalent variables and the metadata, naming, missing-value, and rounding conventions in [AGENTS.md](../AGENTS.md); derive source selectors from the file and its index.
3. Run `uv run main <dataset-id> update-template` and the template checks listed in AGENTS.md.
4. Exercise the variable in the dataset's local backfill and operational-update integration test and inspect the values. Revert these temporary test changes afterward.
5. For proxied virtual sources, confirm the proxy serves the new variable before publishing references to it.
6. Merge the reviewed template changes, then [backfill the variable alone](backfill.md), [validate](validation.md), and publish through the shared guide.
