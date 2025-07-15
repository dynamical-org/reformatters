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
* **Variant** - the specific subset and structure of data from the model. e.g. forecast, analysis, climatology. Variant may include any other information needed to distinguish datasets from the same model.
* **Dataset** - a specific provider-model-variant. e.g. noaa-gfs-forecast



## Integration steps

Before getting started, follow the brief setup steps in README.md > Local development > Setup.

### 1. Initialize a new integration

```bash
uv run main initialize-new-integration <provider> <model> <variant>
```
This will add a number of files within `src/reformatters/<provider>/<model>/<variant>` and `tests/<provider>/<model>/<variant>`.
Please use lower case for the provider, model, and variant. Hyphens are acceptable for the provider
or model (e.g. `icon-eu`).
These files will contain placeholder implementations of the subclasses referenced above. Follow the rest of this doc for guidance 
on how to complete the implementations to integrate your new dataset.

### 2. Register your dataset

Add an instance of your `DynamicalDataset` subclass to the `DYNAMICAL_DATASETS` constant in `src/reformatters/__main__.py`:
```python
from reformatters.provider.model.variant import ProviderModelVariantDataset

DYNAMICAL_DATASETS = [
    ...,
    ProviderModelVariantDataset(storage_config=SourceCoopDatasetStorageConfig()),
]
```
If you plan to write this dataset to a location not maintained by dynamical.org, you can instantiate and pass your own `StorageConfig`, contact feedback@dynamical.org for support.

### 3. Implement `TemplateConfig` subclass

Work through `src/reformatters/$DATASET_PATH/template_config.py`, setting the attributes and method definitions to describe the structure of your dataset.

Hint: providing an AI/LLM with 1) the example template config code to edit, 2) output of running `gdal-info <example source data file>` and 3) any dataset documentation will help it give you a decent first implementation of your `TemplateConfig` subclass.

Using the information in the `TemplateConfig`, `reformatters` writes the Zarr metadata for your dataset to `src/reformatters/$DATASET_PATH/templates/latest.zarr`.  Run this command in your terminal to create or update the template based on the your `TemplateConfig` subclass:
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
uv run main $DATASET_ID backfill-local <append_dim_end> --filter-variable-names <data var name>
```
Reformatting locally can be slow. Choosing an `<append_dim_end>` not long after your template's `append_dim_start` and selecting a single variable to process with `--filter-variable-names` can limit the amount of work.


### 5. Implement `DynamicalDataset` subclass

To operationalize your dataset and have the `update` and `validate` Kubernetes cron jobs be deployed automatically by GitHub CI, implement the two methods in `src/reformatters/$DATASET_PATH/dynamical_dataset.py`

In `dynamical_dataset_test.py` create a test that runs `backfill_local` followed by `update` for a couple data variables.
```bash
uv run pytest tests/$DATASET_PATH/dynamical_dataset_test.py
```


### 6. Deployment

The details here depend on the computing resources and the Zarr storage location you'll be using. Get in touch with feedback@dynamical.org for support at this point if you haven't already.


1. Run a backfill on your local computer: `DYNAMICAL_ENV=prod uv run main $DATASET_ID backfill-local <append-dim-end>`. If this is fast enough and you have the disk space, it is a nice and simple approach.
1. Run a backfill on a kubernetes cluster:
    - This supports parallelism across servers to process much larger datasets.
    - Complete the steps in README.md > Deploying to the cloud > Setup.
    - `DYNAMICAL_ENV=prod uv run main $DATASET_ID backfill-kubernetes <append-dim-end> --max-parallelism N`, then track the job with `kubectl get jobs`.
1. See operational cronjobs in your kubernetes cluster and check their schedule: `kubectl get cronjobs`.
1. To enable issue reporting and cron monitoring with the error reporting service Sentry, create a secret in your kubernetes cluster with your Sentry account's DSN: `kubectl create secret generic sentry --from-literal='DYNAMICAL_SENTRY_DSN=xxx'`.
