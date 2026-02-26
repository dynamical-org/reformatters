# Dataset Integration Guide

Integrate a dataset to reformat into Zarr.

## Overview

Integrating a dataset in dynamical.org `reformatters` is done by subclassing a trio of base classes, customizing their behavior based on the unique characteristics of your dataset.

There are three core base classes to subclass.

1. `TemplateConfig` defines the dataset **structure**.
1. `RegionJob` defines the **process** by which a region of that dataset is reformatted: **downloading, reading, rewriting.**
1. `DynamicalDataset` brings together a `TemplateConfig` and `RegionJob` and defines the compute resources to operationally update and validate a dataset.

### Words

- **Provider** - the agency or organization that publishes the source data. e.g. ECMWF
- **Model** - the model or system that produced the data. e.g. GFS
- **Variant** - the specific subset and structure of data from the model. e.g. forecast, analysis, climatology. Variant may include any other information needed to distinguish datasets from the same model.
- **Dataset** - a specific provider-model-variant. e.g. noaa-gfs-forecast

## Integration steps

Before getting started, follow the brief setup steps in README.md > Local development > Setup.

### Find a reference dataset

Before writing any code, identify the most similar existing dataset in the repo and use it as your primary reference. For example, if you're adding a new ECMWF forecast dataset, study the IFS ENS or AIFS implementations. Look at:

- **Shared provider utilities** — check `src/reformatters/<provider>/` for shared modules (e.g., `ecmwf/ecmwf_grib_index.py`, `ecmwf/ecmwf_config_models.py`). Use these rather than writing your own.
- **Common utilities** — check `src/reformatters/common/iterating.py` for helpers like `group_by`, `item`, `digest` before implementing equivalent logic.
- **Variable names and metadata** — cross-reference variables that already exist in other datasets. If another dataset already has `temperature_2m` or `precipitable_water_atmosphere`, match those names and metadata exactly. Run `uv run pytest tests/common/datasets_cf_compliance_test.py -x` to catch naming inconsistencies early.
- **Kubernetes resource values** — copy resource values (cpu, memory, shared_memory, ephemeral_storage) from a similar dataset rather than guessing. The chunk/shard layout tool (see below) reports a shared memory estimate you can use as a starting point.

### 0. Explore the source dataset

Explore the source dataset to understand the nuances of what's available and how to access it. See [docs/source_data_exploration_guide.md](source_data_exploration_guide.md).

### 1. Initialize a new integration

```bash
uv run main initialize-new-integration <provider> <model> <variant>
```

Provider, model and variant can contain letters, numbers and dashes (e.g. ICON-EU or analyisis-hourly). Capitalization will be normalized for you.

This will add a number of files within `src/reformatters/<provider>/<model>/<variant>` and `tests/<provider>/<model>/<variant>`.

These files will contain placeholder implementations of the subclasses referenced above. Follow the rest of this doc for guidance
on how to complete the implementations to integrate your new dataset.

### 2. Register your dataset

Add an instance of your `DynamicalDataset` subclass to the `DYNAMICAL_DATASETS` constant in `src/reformatters/__main__.py`:

```python
from reformatters.provider.model.variant import ProviderModelVariantDataset

DYNAMICAL_DATASETS = [
    ...,
    ProviderModelVariantDataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig(),
        replica_storage_configs=[ProviderModelIcechunkAwsOpenDataDatasetStorageConfig()],
]
```

If you plan to write this dataset to a location not maintained by dynamical.org, you can instantiate and pass your own `StorageConfig`, contact feedback@dynamical.org for support.

### 3. Implement `TemplateConfig` subclass

Work through `src/reformatters/$DATASET_PATH/template_config.py`, setting the attributes and method definitions to describe the structure of your dataset.

Providing an AI with 1) the example template config code to edit, 2) output of running `gdalinfo <example source data file>` and 3) any dataset documentation will help it give you a decent first implementation of your `TemplateConfig` subclass.

#### Variable naming

Variable names, units, short_name, long_name, and standard_name must be consistent across all datasets in the repo. Before defining a variable, search existing `template_config.py` files for the same physical quantity. If another dataset already defines `temperature_2m`, use the same name and metadata — don't rename based on how the source data labels it. CLAUDE.md has the full metadata conventions.

#### Chunk and shard layout

Run the [chunk/shard layout tool](./chunk_shard_layout_tool.md) in `--search` mode to find chunk and shard sizes — don't estimate by hand. The tool outputs ready-to-paste Python code for `var_chunks` and `var_shards`. Target compressed shard sizes of 100-600 MB. For forecast datasets, keep exactly 1 init_time per shard.

Using the information in the `TemplateConfig`, `reformatters` writes the Zarr metadata for your dataset to `src/reformatters/$DATASET_PATH/templates/latest.zarr`. Run this command in your terminal to create or update the template based on the your `TemplateConfig` subclass:

```bash
uv run main $DATASET_ID update-template
git add src/reformatters/$DATASET_PATH/templates/latest.zarr
```

Tracking the template in git lets us review diffs of any changes to the structure of our dataset.

Run the tests, making any changes necessary.

```bash
uv run pytest tests/$DATASET_PATH/template_config_test.py
```

### 4. Implement `RegionJob` subclass

Work through `src/reformatters/$DATASET_PATH/region_job.py`, implementing the attributes and method definitions based on the unique structure and processing required for your dataset.

There are four required methods:

- `generate_source_file_coords` lists all the files of source data that will be processed to complete the `RegionJob`.
- `download_file` retrieves a specific source file and writes it to local disk.
- `read_data` loads data from a local path and returns a numpy array.
- `operational_update_jobs` is a factory method that returns the `RegionJob`s necessary to update the dataset with the latest available data. You can skip this until you're ready to implement dataset updates, a dataset backfill can be run with just the first three methods.

There are a few optional, additional methods which are described in the example code. Implement them if required for your dataset, otherwise remove them to use the base class `RegionJob` implementations.

Before writing download/read logic from scratch, check if the provider already has shared utilities you should use. For example, ECMWF datasets should use `ecmwf_grib_index.get_message_byte_ranges_from_index()` for byte-range downloads from GRIB index files, not a custom implementation. For `source_groups()`, use `group_by()` from `common.iterating`.

If downloading files from an HTTP/S3 source that provides an index file, set `max_vars_per_download_group` to batch multiple variables per download (e.g. 5-10) rather than downloading one variable at a time. This reduces HTTP overhead significantly.

#### RegionJob tests

Write a unit test for each overridden method, mocking network calls. At minimum test:
- `generate_source_file_coords` — correct count and structure of coords for a small template slice
- `download_file` — correct URL construction and byte ranges (mock the HTTP call)
- `read_data` — correct band selection and dtype (mock the file reader)
- `operational_update_jobs` — returns jobs with correct time range
- `source_groups` — correct grouping if overridden

Generally don't implement integration style tests that make network requests in your `region_job_test.py`, we'll do those in the `dynamical_dataset_test.py`.

```bash
uv run pytest tests/$DATASET_PATH/region_job_test.py
```

You've reached the point where you can run the reformatter locally!

```bash
uv run main $DATASET_ID backfill-local <append_dim_end> --filter-variable-names <data var name>
```

Reformatting locally can be slow. Choosing an `<append_dim_end>` not long after your template's `append_dim_start` and selecting a single variable to process with `--filter-variable-names` can limit the amount of work.

### 5. Implement `DynamicalDataset` subclass

To operationalize your dataset and have the `update` and `validate` Kubernetes cron jobs be deployed automatically by GitHub CI, implement the two methods in `src/reformatters/$DATASET_PATH/dynamical_dataset.py`.

#### Kubernetes resource values

Copy resource values (cpu, memory, shared_memory, ephemeral_storage, pod_active_deadline) from a similar existing dataset rather than guessing. The chunk/shard layout tool's "shared memory" output gives the size of one full shard in memory — use this to inform `shared_memory`. The update cron schedule should run shortly after the source data is expected to be available.

#### Storage configuration

Register your dataset in `__main__.py` with both a primary storage config and a replica. See existing entries in `DYNAMICAL_DATASETS` for the pattern. Create a provider-specific `IcechunkAwsOpenDataDatasetStorageConfig` subclass for the replica following the naming convention of existing ones.

#### Integration test with snapshot values

In `dynamical_dataset_test.py` create a test that runs `backfill_local` followed by `update` for a couple data variables. Include snapshot value assertions — check specific known values at specific coordinates (e.g. `assert float(point["temperature_2m"]) == 28.75`). Snapshot values catch silent regressions in data reading, unit conversion, or coordinate alignment that other tests miss.

```bash
uv run pytest tests/$DATASET_PATH/dynamical_dataset_test.py
```

### Pre-submission checks

Before opening a PR, run these tests to catch cross-dataset issues:

```bash
uv run pytest tests/common/common_template_config_subclasses_test.py tests/common/datasets_cf_compliance_test.py
```

These verify that your template matches the on-disk Zarr, that variable metadata follows CF conventions, and that variable names/units are consistent with other datasets in the repo. Inconsistencies caught here are much easier to fix than after review.

### 6. Deploy

The details here depend on the computing resources and the Zarr storage location you'll be using. Get in touch with feedback@dynamical.org for support at this point if you haven't already.

1. Run a backfill on your local computer: `DYNAMICAL_ENV=prod uv run main $DATASET_ID backfill-local <append-dim-end>`. If this is fast enough and you have the disk space, it is a nice and simple approach.
1. Run a backfill on a kubernetes cluster:
   - This supports parallelism across servers to process much larger datasets.
   - Complete the steps in README.md > Deploying to the cloud > Setup.
   - `DYNAMICAL_ENV=prod uv run main $DATASET_ID backfill-kubernetes <append-dim-end> <jobs-per-pod> <max-parallelism>`, then track the job with `kubectl get jobs`.
1. See operational cronjobs in your kubernetes cluster and check their schedule: `kubectl get cronjobs`.
1. To enable issue reporting and cron monitoring with the error reporting service Sentry, create a secret in your kubernetes cluster with your Sentry account's DSN: `kubectl create secret generic sentry --from-literal='DYNAMICAL_SENTRY_DSN=xxx'`.

## 7. Validate

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

## 8. Update dataset catalog documentation

Update the dataset catalog docs on `dynamical.org` by adding entries into the `catalog.js`, rebuilding (`npm run build`), and merging updates to main in `https://github.com/dynamical-org/dynamical.org`.