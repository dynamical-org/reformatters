This project contains code to reformat weather data into the zarr v3 file format.

## Approach overview

Datasets are created in 3 phases:
1. A template of the dataset, in the form of zarr metadata that is checked into the repo, is created with `uv run main <dataset_id> update-template`. This template (not in-code config) is loaded by steps 2 and 3 and drives processing and output in those steps. This approach of checking in the metadata allow us to review diffs if the structure or metadata of the dataset changes.
2. A zarr backfill is run. The backfill uses kubernetes indexed jobs to run work in parallel. When the user runs a `uv run main <dataset-id> backfill-kubernetes ...` command the metadata for the zarr is first written by the local process to the final zarr store, then a kubernetes index job is kicked off with each job index responsible for writing a portion of the zarr chunk data into the zarr archive.
3. Operational updates to the zarr are run using a kubernetes cronjob and validated by another kubernetes cronjob which runs after the update is expected to succeed. To ensure the archive is valid to readers throughout the update, the update writes data chunks for all data variables first, then updates the zarr metdata to reflect the larger dataset size. The operational update runs a single process to avoid interprocess communication while ensuring the metadata is updated last.

Common utilities and conventions seek to reduce the amount of unique code required for a single source of weather data/zarr dataset.

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
│   ├── validation.py        # Dataset validators
│   └── ...                  # Other utilities
├── <provider>/              # e.g., noaa/, ecmwf/, dwd/
│   └── <model>/             # e.g., gfs/, hrrr/, icon_eu/
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
docs/                        # Additional documentation
deploy/                      # Docker and kubernetes configs
```

## Core classes

Integrating a dataset requires subclassing three base classes. For complete details, see [docs/dataset_integration_guide.md](docs/dataset_integration_guide.md).

### TemplateConfig (`src/reformatters/common/template_config.py`)

Defines the **structure** of a dataset: dimensions, coordinates, data variables, and their attributes/encodings. Key responsibilities:
- Declare `dims`, `append_dim`, `append_dim_start`, `append_dim_frequency`
- Implement `dataset_attributes`, `coords`, `data_vars`, `dimension_coordinates()`, and optionally `derive_coordinates()`
- Generates and persists zarr metadata to `templates/latest.zarr` via `update_template()`

### RegionJob (`src/reformatters/common/region_job.py`)

Defines the **process** for reformatting a region of the dataset: downloading, reading, and writing data. Key responsibilities:
- Implement `generate_source_file_coords()` - list source files needed for a region
- Implement `download_file()` - retrieve a source file to local disk
- Implement `read_data()` - load data from a local file into a numpy array
- Implement `operational_update_jobs()` (class method) - factory for operational update jobs
- Optionally override `source_groups()`, `apply_data_transformations()`, `update_template_with_results()`

A `RegionJob` processes a slice of the append dimension (e.g., one shard of `init_time`) for a group of data variables.

### DynamicalDataset (`src/reformatters/common/dynamical_dataset.py`)

**Brings together** a `TemplateConfig` and `RegionJob` class, plus storage and operational configuration. Key responsibilities:
- Declare `template_config` and `region_job_class`
- Configure `primary_storage_config` and optional `replica_storage_configs`
- Implement `operational_kubernetes_resources()` - define update/validate cron jobs
- Implement `validators()` - return validation functions for the dataset

### How they relate

```
DynamicalDataset
├── template_config: TemplateConfig  # Dataset structure
├── region_job_class: RegionJob      # Processing logic
└── storage configs                  # Where to write data

TemplateConfig.update_template() → templates/latest.zarr (checked into git)
                                         ↓
RegionJob.get_jobs() → list[RegionJob]  (each job = region × variables)
                                         ↓
RegionJob.process() → download → read → transform → write shards
```

## CLI commands

Run via `uv run main` (alias for `uv run python -m reformatters`).

### Global commands
- `uv run main --help` - show all commands and registered datasets
- `uv run main initialize-new-integration <provider> <model> <variant>` - scaffold new dataset
- `uv run main deploy` - deploy operational kubernetes resources for all datasets

### Per-dataset commands

Each dataset (e.g., `noaa-gfs-forecast`) exposes these subcommands:

| Command | Description |
|---------|-------------|
| `update-template` | Generate zarr metadata from `TemplateConfig` and write to `templates/latest.zarr` |
| `backfill-local <append_dim_end>` | Run reformatting locally (dev/test only) |
| `backfill-kubernetes <append_dim_end> <jobs_per_pod> <max_parallelism>` | Launch kubernetes indexed job for parallel backfill |
| `process-backfill-region-jobs` | Internal: called by kubernetes workers to process assigned jobs |
| `update` | Operational update: extend dataset with latest available data |
| `validate` | Run validators on the dataset |

Example:
```bash
# Update template metadata
uv run main noaa-gfs-forecast update-template

# Local backfill (for development)
uv run main noaa-gfs-forecast backfill-local 2024-01-02 --filter-variable-names temperature_2m

# Kubernetes backfill (production)
DYNAMICAL_ENV=prod uv run main noaa-gfs-forecast backfill-kubernetes 2024-06-01 10 100
```

## Parallelization model

Kubernetes indexed jobs provide parallelism for backfills. Each worker computes the full job list, then selects its subset.

### How it works

1. **Job generation**: `RegionJob.get_jobs()` creates all jobs by combining:
   - Regions: slices along the append dimension (typically one shard each)
   - Variable groups: subsets of data variables that share source files

2. **Worker assignment**: `get_worker_jobs(all_jobs, worker_index, workers_total)` distributes jobs round-robin:
   ```python
   # Worker 0 of 3: jobs[0], jobs[3], jobs[6], ...
   # Worker 1 of 3: jobs[1], jobs[4], jobs[7], ...
   # Worker 2 of 3: jobs[2], jobs[5], jobs[8], ...
   return tuple(islice(jobs, worker_index, None, workers_total))
   ```

3. **Kubernetes execution**: Each pod receives `WORKER_INDEX` and `WORKERS_TOTAL` env vars. `process-backfill-region-jobs` calls `get_jobs()` with these values to compute which jobs to run.

### Key insight
Every worker independently computes the same ordered list of all jobs, then deterministically selects its subset. No coordinator or job queue is needed.

## Tools
* `uv` to manage dependencies and python environments
* `ruff` for linting and formatting
* `mypy` for type checking
* `pytest` for testing
* `pre-commit` to automatically lint and format as you git commit
* `docker` to package the code and dependencies
* `kubernetes` indexed jobs to run work in parallel and cronjobs to run ongoing dataset updates and validation

## Development commands
* Add dependency: `uv add <package> [--dev]`. Use `--dev` to add a development only dependency.
* Lint: `uv run ruff check`
* Type check: `uv run mypy`
* Format: `uv run ruff format`
* Test: `uv run pytest`
* Fast/unit tests: `uv run pytest -m "not slow"`
* Single test: `uv run pytest tests/test_file.py::test_function_name`
* The aider lint-cmd (`uv run ruff format && uv run ruff check --fix && uv run mypy`) will automatically run for you.
* Use `uv run ...` to run python commands in the environment, e.g. `uv run main --help`

## Code Style
* Write code that explains itself rather than needs comments
* Add only extremely minimal code comments and no docstrings unless I ask for them, but don't remove existing comments
  * Add comments only when doing things out of the ordinary, to highlight gotchas, or if less clear code is required due to an optimization
* Use Python 3.13+ features
* Follow mypy strict mode. If you need to add an ignore, ignore a specific check like `# type: ignore[specific]`. Always annotate types on all function arguments and return types.
* Try to follow ruff format
* Don't write error handing code unless I ask for it, nor smooth over exceptions/errors unless they are expected as part of control flow. In general, write code that will raise an exception early if something isn't expected. Enforce important expectations with asserts.
* Test each module with pytest
