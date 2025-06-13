# Dataset Integration Guide
Integrate a dataset to reformat into Zarr.

## Overview
Integrating a dataset in dynamical.org `reformatters` is done by subclassing a trio of base classes, customizing their behavior based on the unique characteristics of your dataset.

There are three core base classes to subclass.
1. `TemplateConfig` defines the dataset **structure**.
1. `RegionConfig` defines the **process** by which a region of that dataset is reformatted: **downloading, reading, rewriting.**
1. `DynamicalDataset` brings together a `TemplateConfig` and `RegionJob` and defines the compute resources to operationally update and validate a dataset.

### Words
* **Provider** - the agency or organization that publishes the source data. e.g. ECMWF
* **Model** - the model or system that produced the data. e.g. GFS
* **Variant** - the specific subset and structure of data from the model. e.g. forecast, analysis, climatology. Variant may include any other information needed to distinquish datasets from the same model.
* **Dataset** - a specific provider-model-variant. e.g. noaa-gfs-forecast



## Integration steps

### 1. Set up template

#### Copy template
From a terminal in the root of this repository, run these commands to copy a template of dataset integration code and tests.
```bash
DATASET_ID="<provider>-<model>-<variant>"
DATASET_PATH="<provider>/<model>/<variant>"
mkdir -p src/reformatters/$DATASET_PATH
cp -r src/reformatters/example/ src/reformatters/$DATASET_PATH
mkdir -p tests/$DATASET_PATH
cp -r tests/example tests/$DATASET_PATH
```

#### Rename
Run the following find/replaces. Follow PEP 8 on abbreviation capitalization in class names, e.g. `NoaaGfsForecastDataset`.

Find and replace within `src/reformatters/$DATASET_PATH` and `tests/$DATASET_PATH`:
1. `ExampleDataset` -> `<Producer><Model><Variant>Dataset`
1. `ExampleTemplateConfig` -> `<Producer><Model><Variant>TemplateConfig`
1. `ExampleRegionJob` -> `<Producer><Model><Variant>RegionJob`
1. `ExampleDataVar` -> `<Producer>[<model>]DataVar`
1. `ExampleInternalAttrs` -> `<Producer>[<model>]DataVar`
1. `ExampleSourceFileCoord` -> `<Producer><Model>SourceFileCoord`
1. `reformatters.example` -> `reformatters.<producer>.<model>.<variant>` (imports in tests)

DataVar, InternalAttrs, and SourceFileCoord definitions can often be shared among mutiple datasets from the same producer.

### 2. Register your dataset

Add an instance of your `DynamicalDataset` subclass to the `DYNAMICAL_DATASETS` constant in `src/reformatters/__main__.py`:
```python
from reformatters.provider.model.variant import ProviderModelVariantDataset

DYNAMICAL_DATASETS = [
    ...,
    ProviderModelVariantDataset(),
]
```

### 3. Implement `TemplateConfig` subclass

Work through `src/reformatters/$DATASET_PATH/template_config.py`, setting the attributes and method definitions to describe the structure of your dataset.

Hint: providing an AI/LLM with 1) the example template config code to edit, 2) output of running `gdal-info <example source data file>` and 3) any dataset documentation will help it give you a decent first implementation of your `TemplateConfig` subclass.

Using the information in the TemplateConfig, `reformatters` writes the Zarr metadata for your dataset to `src/reformatters/$DATASET_PATH/templates/latest.zarr`.  Run this command in your terminal to create or update the template based on the your `TemplateConfig` subclass:
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
* `generate_source_file_coords` lists all the files of source data that will be processed to complete the `RegionJob`.
* `download_file` retrieves a specific source file and writes it to local disk.
* `read_data` loads data from a local path and returns a numpy array.
* `operational_update_jobs` is a factory method that returns the `RegionJob`s necessary to update the dataset with the latest available data. You can skip this until you're ready to implement dataset updates, a dataset backfill can be run with just the first three methods.

There are a few optional, additional methods which are described in the example code. Implement them if required for your dataset, otherwise remove them to use the base class `RegionJob` implementations.

Write a test or two for any custom logic you've created. Generally don't implement integration style tests that make network requests in your `region_job_test.py`, we'll do those in the `dynamical_dataset_test.py`.
```bash
uv run pytest tests/$DATASET_PATH/region_job_test.py
```

You've reached the point where you can run the reformatter locally!
```bash
uv run main $DATASET_ID reformat-local <append_dim_end> --filter-variable-names <data var name>
```
Reformatting locally can be slow. Choosing an `<append_dim_end>` not long after your template's `append_dim_start` and selecting a single variable to process with `--filter-variable-names` can limit the amount of work.


### 5. Implement `DynamicalDataset` subclass

To operationalize your dataset and have the `update` and `validate` Kubernetes cron jobs be deployed automatically by GitHub CI, implement the two methods in `src/reformatters/$DATASET_PATH/dynamical_dataset.py`

In `dynamical_dataset_test.py` create a test that runs `reformat_local` followed by `reformat_operational_update` for a couple data variables.
```bash
uv run pytest tests/$DATASET_PATH/dynamical_dataset_test.py
```


### 6. Deployment

The details here depend on the computing resources and the Zarr storage location you'll be using. Get in touch with the dynamical.org team for support at this point if you haven't already.

Complete the steps in README.md > Deploying to the cloud > Setup.

1. Run a backfill: `DYNAMICAL_ENV=prod uv run main $DATASET_ID reformat-kubernetes <append-dim-end> --max-parallelism N`.
1. Make sure CI deployed the operational cronjobs and check their schedule: `kubectl get cronjobs`