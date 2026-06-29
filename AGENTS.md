This project contains code to reformat weather data into the Zarr v3 / Icechunk file format.

## Approach overview

Datasets are created in 3 phases:
1. A template of the dataset, in the form of zarr metadata that is checked into the repo, is created with `uv run main <dataset_id> update-template`. This template (not in-code config) is loaded by steps 2 and 3 and drives processing and output in those steps. This approach of checking in the metadata allow us to review diffs if the structure or metadata of the dataset changes.
2. A zarr backfill is run. The backfill uses kubernetes indexed jobs to run work in parallel. When the user runs a `uv run main <dataset-id> backfill-kubernetes ...` command the metadata for the zarr is first written by the local process to the final zarr store, then a kubernetes index job is kicked off with each job index responsible for writing a portion of the zarr chunk data into the zarr archive.
3. Operational updates to the zarr are run using a kubernetes cronjob and validated by another kubernetes cronjob which runs after the update is expected to succeed. Updates use the same parallel worker model as backfills. To ensure the archive is valid to readers throughout the update, zarr v3 metadata is written only after all workers finish, and icechunk stores use a temporary branch that is atomically merged to main.

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
docs/
├── dataset_integration_guide.md      # Step-by-step new dataset integration walkthrough
├── parallel_processing.md            # How parallel writes coordinate across workers
├── virtual_datasets.md               # Writing + reading virtual (chunk reference) Icechunk datasets
├── add_new_variable.md               # Add new variable to an existing dataset
├── validation.md                     # Run + read validation plots; data quality checklist
├── chunk_shard_layout_tool.md        # Zarr V3 chunk/shard layout optimizer
├── source_data_exploration_guide.md  # Explore/document source data structure before integration
├── ops_card.md                       # Operations: monitoring, troubleshooting, manual updates
└── staging.md                        # Run concurrent dataset versions for testing
deploy/                      # Docker and kubernetes configs
├── Dockerfile               # Container image for reformatter jobs
└── aws/                     # nodepool.yaml, create_new_aws_open_data_bucket.sh
```

- **Shared provider utilities** Check `src/reformatters/<provider>/` for shared modules (e.g., `ecmwf/ecmwf_grib_index.py`, `noaa/noaa_utils.py`).
- **Common utilities** Look for relevant utilities in `src/reformatters/common/` before implementing equivalent logic. An incomplete list: `download.py` (`http_download_to_disk`), `iterating.py` (`group_by`, `item`, `digest`), `logging.py` (`get_logger`), `pydantic.py` (`replace`, `FrozenBaseModel`), `retry.py`, `time_utils.py` (`whole_hours`).

## Core classes

Integrating a dataset requires subclassing three base classes. For a step by step walkthrough, see [docs/dataset_integration_guide.md](docs/dataset_integration_guide.md) and for complete details of what and how subclassers should implement see the commented templates in `src/reformatters/example/{dynamical_dataset|template_config|region_job}.py`.

### TemplateConfig
Base class: `src/reformatters/common/template_config.py`, commented example subclass: `src/reformatters/example/template_config.py`.

Defines the **structure** of a dataset: dimensions, coordinates, data variables, and their attributes/encodings. Key responsibilities:
- Declare `dims`, `append_dim`, `append_dim_start`, `append_dim_frequency`. `dims` is keyed by group: `dims = {ROOT: (...)}` for a single-level dataset, plus one entry per vertical group (e.g. `"pressure_level": (..., "pressure_level")`).
- Implement `dataset_attributes`, `coords`, `data_vars`, `dimension_coordinates()`, and optionally `derive_coordinates()`
- Generates and persists zarr metadata to `templates/latest.zarr` via `update_template()`. The template is an `xarray.DataTree` (one node per group; a single-level dataset is a one-node tree); `get_template()` returns that DataTree.

Always regenerate the template after any metadata changes with `uv run main <dataset-id> update-template`.

Run these tests after updating a template: `uv run pytest tests/common/common_template_config_subclasses_test.py tests/common/datasets_cf_compliance_test.py`.

#### Metadata conventions
Metadata attributes for variables and coordinates must follow CF Conventions.
The `standard_name` and `units` fields must match CF definitions if one exists for that variable; if one doesn't, use SI `units` and leave `standard_name` unset.
Use ECMWF variable name for `long_name` and ECMWF short name for `short_name`.  When adding a variable, search to see if another dataset already has an equivalent variable (e.g. `temperature_2m`), match those names and metadata exactly.


### RegionJob
Base class: `src/reformatters/common/region_job.py`, commented example subclass: `src/reformatters/example/region_job.py`.

Defines the **process** for reformatting a region of the dataset: downloading, reading, and writing data. `RegionJob` is the shared base (partitioning via `get_jobs()`, `generate_source_file_coords()`, `operational_update_jobs()`, `update_template_with_results()`); `MaterializedRegionJob(RegionJob)` adds the concrete download → read → write pipeline (`download_file()`, `read_data()`, `apply_data_transformations()`, `process()` and the parallelism tunables). Materialized (rechunked) datasets subclass `MaterializedRegionJob`. Key responsibilities:
- Implement `generate_source_file_coords()` - list source files needed for a region
- Implement `download_file()` - retrieve a source file to local disk
- Implement `read_data()` - load data from a local file into a numpy array
- Implement `operational_update_jobs()` (class method) - factory for operational update jobs
- Optionally override `source_file_var_groups()` (vars that share a source file; a single source file may span vertical groups), `get_processing_region()` (buffer processed region to support deaccumulation & interpolation), `apply_data_transformations()` (deaccumulation, etc.), `update_template_with_results()` (trim based on actual data processed).

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

Vertical levels: Single-level and surface variables live at the dataset root with the level encoded in the variable name (e.g. `temperature_2m`, `pressure_surface`). A variable available on a dense, comparable set of vertical levels does not have a level suffix and instead lives in a zarr group named after its vertical dimension — the group name and dimension name are the same (`pressure_level`, `model_level`; others may be added as needed), e.g. `pressure_level/temperature` is a variable with dimensions (time, latitude, longitude, pressure_level). Dimension coordinates shared with the root (time, lead time, latitude/longitude, ensemble_member, spatial_ref) are duplicated into each group so a group can be opened on its own. A group with no variables is omitted. A variable's group is a per-variable property (it sets the variable's zarr path and dims), not a job boundary. For the design — type scheme, group-keyed `dims`, structural validators, and DataTree representation — see [docs/virtual_datasets.md](docs/virtual_datasets.md#multi-group-vertical-datasets) (multi-group is currently used only by virtual datasets).

Spatial dimensions: If the source data uses a geographic projection we use dimensions latitude and longitude, else y and x are used for projected datasets.

Latitude ordering: latitude/y is stored north-first — descending latitude, index 0 is the northernmost row — matching GDAL/GIS and the common analysis-ready convention (ERA5, GFS). Raw GRIB scan direction is not consistent across sources (it is set per message; e.g. GFS scans north-first, HRRR south-first), so a source whose native scan is south-first is flipped on decode to conform. For a virtual (byte-reference) dataset that flip lives in the per-variable serializer — the only place its data can be reordered — so it requires decoder support; a virtual dataset from a south-first source must not be published until it conforms.

## CLI commands

Run via `uv run main`.

### Global commands
- `uv run main --help` - Show all commands and registered datasets
- `uv run main initialize-new-integration <provider> <model> <variant>` - Scaffold new dataset
- `uv run main <dataset-id> update-template` - Regenerate `templates/latest.zarr`. Run this after any change to a `TemplateConfig` subclass's metadata.

## Parallelization model

Both backfills and operational updates distribute work across Kubernetes indexed jobs. Every worker independently computes the same ordered list of all jobs, then deterministically selects its subset via round-robin. No coordinator or job queue is needed.

1. **Job generation**: `RegionJob.get_jobs()` creates all jobs by combining regions (shard slices along append dim) × variable groups (controlled by `max_vars_per_job`), with optional filters.

2. **Worker assignment**: `get_worker_jobs(all_jobs, worker_index, workers_total)` distributes jobs round-robin.

3. **Coordination**: Workers coordinate via files in `_internal/{job_name}/` in object store. Worker 0 does setup, all workers process, the last worker (by index) finalizes.

4. **Reader safety**: Zarr v3 stores defer metadata writes until finalization. Icechunk stores use a temporary branch so readers on `main` never see partial data.

See [docs/parallel_processing.md](docs/parallel_processing.md) for details on coordination protocol, failure modes, and retry behavior.

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
  * In non-test code, the vast majority of comments should be one line. State the non-obvious fact the next reader needs, not the reasoning behind the change: no failure-mode stories, no defending why the code is correct, no operational instructions. Point to docs (e.g. "..., see docs/parallel_processing.md.") to add richer context only for complex topics.
  * In non-test code, an assert or validator with a clear message is its own documentation — don't add a comment restating what it enforces or what would break without it.
* Use Python 3.13+ features
* Follow ty type checking. If you need to add an ignore, ignore a specific check like `# ty: ignore[specific]`. Always annotate types on all function arguments and return types.
* Follow ruff format
* Test each module with pytest
* Log don't print: `from reformatters.common.logging import get_logger` and `log = get_logger(__name__)`
* Keep documentation up to date. After making a change that would update any .md file in the repo (CLAUDE.md, docs/*, etc.), always update relevant docs, regardless of if you have been explicitly asked.
* CLAUDE.md is an alias for AGENTS.md