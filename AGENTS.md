This project contains code to reformat weather data into the Zarr v3 / Icechunk file format.

## Approach overview

Datasets are created in 3 phases:
1. A template of the dataset, in the form of zarr metadata that is checked into the repo, is created with `uv run main <dataset_id> update-template`. This template (not in-code config) is loaded by steps 2 and 3 and drives processing and output in those steps. This approach of checking in the metadata allow us to review diffs if the structure or metadata of the dataset changes.
2. A zarr backfill is run. The backfill uses kubernetes indexed jobs to run work in parallel. When the user runs a `uv run main <dataset-id> backfill-kubernetes ...` command the metadata for the zarr is first written by the local process to the final zarr store, then a kubernetes index job is kicked off with each job index responsible for writing a portion of the zarr chunk data into the zarr archive.
3. Operational updates to the zarr are run using a kubernetes cronjob and validated by another kubernetes cronjob which runs after the update is expected to succeed. To ensure the archive is valid to readers throughout the update, the update writes data chunks for all data variables first, then updates the zarr metdata to reflect the larger dataset size. The operational update runs a single process to avoid interprocess communication while ensuring the metadata is updated last.

## Repository structure

```
src/reformatters/
├── __main__.py              # CLI entrypoint, dataset registry
├── common/                  # Shared utilities and base classes
│   ├── dynamical_dataset.py # DynamicalDataset base class
│   ├── template_config.py   # TemplateConfig base class
│   ├── region_job.py        # RegionJob base class
│   ├── config_models.py     # DataVar, Coordinate, etc.
│   ├── iterating.py         # Parallelization helpers (get_worker_jobs)
│   ├── kubernetes.py        # Job/CronJob definitions
│   ├── storage.py           # Storage config and store factories
│   ├── validation.py        # Dataset validators run operationally
│   └── ...                  # Other utilities
├── <provider>/              # e.g., noaa/, ecmwf/, dwd/
│   └── <model>/             # e.g., gfs/, hrrr/, ifs_ens/
│       └── <variant>/       # e.g., forecast/, analysis/
│           ├── __init__.py
│           ├── dynamical_dataset.py
│           ├── template_config.py
│           ├── region_job.py
│           └── templates/
│               └── latest.zarr/  # Checked-in zarr metadata
├── contrib/                 # Community-contributed datasets
└── example/                 # Template for new integrations

tests/                       # Mirrors src/ structure
docs/                        # Documentation
deploy/                      # Docker and kubernetes configs
```

## Core classes

Integrating a dataset requires subclassing three base classes. For a step by step walkthrough, see [docs/dataset_integration_guide.md](docs/dataset_integration_guide.md) and for complete details of what and how subclassers should implement see the commented templates in `src/reformatters/example/{dynamical_dataset|template_config|region_job}.py`.

### TemplateConfig
Base class: `src/reformatters/common/template_config.py`, commented example subclass: `src/reformatters/example/template_config.py`.

Defines the **structure** of a dataset: dimensions, coordinates, data variables, and their attributes/encodings. Key responsibilities:
- Declare `dims`, `append_dim`, `append_dim_start`, `append_dim_frequency`
- Implement `dataset_attributes`, `coords`, `data_vars`, `dimension_coordinates()`, and optionally `derive_coordinates()`
- Generates and persists zarr metadata to `templates/latest.zarr` via `update_template()`

Always regenerate the template after any metadata changes with `uv run main <dataset-id> update-template`.

Run these tests after updating a template: `uv run pytest tests/common/common_template_config_subclasses_test.py tests/common/datasets_cf_compliance_test.py`.

#### Metadata conventions
Metadata attributes for variables and coordinates must follow CF Conventions.
The `standard_name` and `units` fields must match CF definitions if one exists for that variable; if one doesn't, use SI `units` and leave `standard_name` unset.
Use ECMWF variable name for `long_name` and ECMWF short name for `short_name`.


### RegionJob
Base class: `src/reformatters/common/region_job.py`, commented example subclass: `src/reformatters/example/region_job.py`.

Defines the **process** for reformatting a region of the dataset: downloading, reading, and writing data. Key responsibilities:
- Implement `generate_source_file_coords()` - list source files needed for a region
- Implement `download_file()` - retrieve a source file to local disk
- Implement `read_data()` - load data from a local file into a numpy array
- Implement `operational_update_jobs()` (class method) - factory for operational update jobs
- Optionally override `source_groups()` (vars that can be accessed together), `get_processing_region()` (buffer processed region to support deaccumulation & interpolation), `apply_data_transformations()` (deaccumulation, etc.), `update_template_with_results()` (trim based on actual data processed).

A `RegionJob` processes a slice of the append dimension (e.g., one shard along `init_time` or `time`) for a group of data variables.

### DynamicalDataset
Base class: `src/reformatters/common/dynamical_dataset.py`, commented example subclass: `src/reformatters/example/dynamical_dataset.py`.

**Brings together** a `TemplateConfig` and `RegionJob` class, plus storage and operational configuration. Key responsibilities:
- Declare `template_config` and `region_job_class`
- Configure `primary_storage_config` and optional `replica_storage_configs`
- Implement `operational_kubernetes_resources()` - define update/validate cron jobs
- Implement `validators()` - return validation functions for the dataset

## Common dataset structures

1. **Forecast dataset** Dimensions init_time, lead_time, latitude/y, longitude/x [, ensemble_member].

2. **Analysis dataset** Dimensions time, latitude/y, longitude/x [, ensemble_member]. When creating an analysis dataset from a forecast archive we take the shortest available lead time, flattening the init_time and lead_time dims into a single time dim.

Vertical levels: Our current datasets include selected vertical levels, which we combine with the variable’s name to create the variable name in the dataset (e.g. temperature_2m). In the future we plan to include model_level and/or pressure_level dimensions for core variables to save their values at all levels.

Spatial dimensions: If the source data uses a geographic projection we use dimensions latitude and longitude, else y and x are used for projected datasets.

## CLI commands

Run via `uv run main`.

### Global commands
- `uv run main --help` - Show all commands and registered datasets
- `uv run main initialize-new-integration <provider> <model> <variant>` - Scaffold new dataset
- `uv run main <dataset-id> update-template` - Regenerate `templates/latest.zarr`. Run this after any change to a `TemplateConfig` subclass's metadata.

## Parallelization model

Kubernetes indexed jobs provide parallelism for backfills. Every worker independently computes the same ordered list of all jobs,
then deterministically selects its subset. No coordinator or job queue is needed.

1. **Job generation**: `RegionJob.get_jobs()` creates all jobs by combining:
   - Regions: slices along the append dimension (typically one shard each)
   - Variable groups: subsets of data variables that share source files
   - Filters: for append dim start/end/contains and variables applied

2. **Worker assignment**: `get_worker_jobs(all_jobs, worker_index, workers_total)` distributes jobs round-robin: `islice(jobs, worker_index, None, workers_total)`

3. **Kubernetes execution**: Each pod receives `WORKER_INDEX` and `WORKERS_TOTAL` env vars.

## Tools
* `uv` to manage pythons and dependencies and run python code
* `ruff` for linting and formatting
* `ty` for type checking
* `pytest` for testing
* `prek` to automatically lint and format as you git commit
* `docker` to package the code and dependencies
* `kubernetes` indexed jobs to run work in parallel and cronjobs to run ongoing dataset updates and validation

## Development commands
* `uv run prek install` to set up the git hooks that will ensure ruff check, ruff format and ty pass.
* Add dependency: `uv add <package> [--dev]`. Use `--dev` to add a development only dependency.
* Lint: `uv run ruff check [--fix]`
* Type check: `uv run ty check --output-format concise`. Remove the output format flag for more detailed errors. `uv run ty check [--flags] specific/dirs or/files.py`
* Format: `uv run ruff format`
* Test: `uv run pytest`
* Fast/unit tests: `uv run pytest -m "not slow"`
* Single test: `uv run pytest tests/test_file.py::test_function_name`
* Important: always run all of these checks before committing, do not skip them: `uv run ruff format && uv run ruff check --fix && uv run ty check`.
* Use `uv run ...` to run python commands in the environment, e.g. `uv run python -c "..."`, `uv run src/scripts/foo.py`. Do not call `python3` when working in this repo.

## Code Style
* Write code that explains itself rather than needs comments.
* Simplicity is paramount. Always look for ways to simplify, use existing utilities and approaches in the code base rather than creating new code, and identify and suggest architectural improvements.
* Don't write error handing code unless I ask for it, nor smooth over exceptions/errors unless they are expected as part of control flow. In general, write code that will raise an exception early if something isn't expected. Enforce important expectations with asserts.
* Add only extremely minimal code comments and no docstrings unless I ask for them, but don't remove existing comments.
  * Add comments only when doing things out of the ordinary, to highlight gotchas, or if less clear code is required due to an optimization.
* Use Python 3.13+ features
* Follow ty type checking. If you need to add an ignore, ignore a specific check like `# ty: ignore[specific]`. Always annotate types on all function arguments and return types.
* Follow ruff format
* Test each module with pytest
