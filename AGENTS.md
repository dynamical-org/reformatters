This project contains code to reformat weather data into the Zarr v3 / Icechunk file format.

## Approach overview

Datasets are created in 3 phases:
1. A template of the dataset, in the form of zarr metadata that is checked into the repo, is created with `uv run main <dataset_id> update-template`. This template (not in-code config) is loaded by steps 2 and 3 and drives processing and output in those steps. This approach of checking in the metadata allow us to review diffs if the structure or metadata of the dataset changes.
2. A zarr backfill is run. The backfill uses kubernetes indexed jobs to run work in parallel. When the user runs a `uv run main <dataset-id> backfill-kubernetes ...` command the metadata for the zarr is first written by the local process to the final zarr store, then a kubernetes index job is kicked off with each job index responsible for writing a portion of the zarr chunk data into the zarr archive.
3. Operational updates to the zarr are run using a kubernetes cronjob and validated by another kubernetes cronjob which runs after the update is expected to succeed. Updates use the same parallel worker model as backfills. To ensure the archive is valid to readers throughout the update, zarr v3 metadata is written only after all workers finish, and icechunk backfills and materialized updates use a temporary branch that is atomically merged to main; virtual updates commit directly to main.

## Repository structure

```
src/reformatters/
├── __main__.py              # CLI entrypoint, dataset registry
├── common/                  # Shared utilities and base classes
│   ├── dynamical_dataset.py # DynamicalDataset base class
│   ├── template_config.py   # TemplateConfig base class
│   ├── region_job.py        # RegionJob base class
│   ├── materialized_region_job.py # MaterializedRegionJob base class
│   ├── virtual_region_job.py # VirtualRegionJob base class
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
├── example_materialized/    # Teaching template for a materialized (time-optimized) dataset
└── example_virtual/         # Teaching template for a virtual (spatial-optimized) dataset

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

Integrating a dataset requires subclassing three base classes. For a step by step walkthrough, see [docs/dataset_integration_guide.md](docs/dataset_integration_guide.md) and for complete details of what and how subclassers should implement see the commented templates in `src/reformatters/example_{materialized|virtual}/{dynamical_dataset|template_config|region_job}.py`.

### TemplateConfig
Base class: `src/reformatters/common/template_config.py`, commented example subclasses: `src/reformatters/example_{materialized|virtual}/template_config.py`.

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
Categorical / flag variables set `flag_values` (the coded values) and `flag_meanings` (a blank separated label per value) per CF Conventions section 3.5. Verify the codes against the authoritative source table for that product (e.g. GRIB2 code table 4.201/4.222, the NSSL MRMS flag tables) rather than guessing.

Comment vs. review note. Put intrinsic, always-true variable facts (quirks, sentinel values, what the variable physically represents if not clear in the name/long_name) in the variable's `comment` attr so they travel with the data — these get no validation-report review note. Most common variables need no `comment` unless their interpretation is unusual. Put time-windowed characteristics of a specific archive (version-boundary behavior changes, historical low-quality windows, source outages) in the validation report's `### Review notes` (see [docs/validation.md](docs/validation.md) §3e) — these get no `comment`, since they would go stale in static template metadata as the archive grows. Each fact lives in exactly one place based on its kind.

#### Encoding conventions

`keep_mantissa_bits` (on a data variable's `internal_attrs`) sets how many of a float32's 23 mantissa bits survive before compression — lower rounds harder and compresses better but coarsens the data; `"no-rounding"` keeps all 23 (used by virtual datasets, whose bytes are never rewritten, and any variable that must stay exact). Pick it per the physical precision the variable needs, not the source's stored precision. Defaults by variable kind:

- **7** — general default for most variables.
- **6** — wind components / speeds.
- **8** — precipitation flux and rates.
- **11** — pressure variables with `units="Pa"`.

Match an existing equivalent variable's value across datasets rather than re-deriving. Existing datasets predate this table and are not all consistent with it (e.g. some carry pressure at 10); leave them as-is unless separately asked to reconcile.


### RegionJob
Base class: `src/reformatters/common/region_job.py`, commented example subclasses: `src/reformatters/example_{materialized|virtual}/region_job.py`.

Defines the **process** for reformatting a region — a slice of the append dim (e.g. one shard along `init_time` or `time`) for a group of data variables. `RegionJob` is the shared base: partitioning (`get_jobs()`), `generate_source_file_coords()` (list a region's source files), and `operational_update_jobs()` (class-method factory for update jobs). Both are required; the rest is subclass-specific.

- **`MaterializedRegionJob`** rechunks and rewrites bytes. Implement `download_file()` (fetch a source file to disk) and `read_data()` (load one variable into a numpy array); the base runs the download → read → transform → write pipeline. Optionally override `source_file_var_groups()` (vars sharing a source file, which may span vertical groups), `get_processing_region()` (buffer for deaccumulation/interpolation), `apply_data_transformations()`, `update_template_with_results()` (trim to processed data).
- **`VirtualRegionJob`** writes chunk references, not bytes. Implement `discover_available()` (which pending files are fetchable now) and `file_refs()` (the byte-range references one file contributes); the base owns the write loop and atomic commits. See [docs/virtual_datasets.md](docs/virtual_datasets.md).

### DynamicalDataset
Base class: `src/reformatters/common/dynamical_dataset.py`, commented example subclasses: `src/reformatters/example_{materialized|virtual}/dynamical_dataset.py`.

**Brings together** a `TemplateConfig` and `RegionJob` class, plus storage and operational configuration. Key responsibilities:
- Declare `template_config` and `region_job_class`
- Configure `primary_storage_config` and optional `replica_storage_configs`
- Implement `operational_kubernetes_resources()` - define update/validate cron jobs
- Implement `validators()` - return validation functions for the dataset

## Dataset structures

1. **Forecast dataset** Dimensions init_time, lead_time, latitude/y, longitude/x [, ensemble_member].

2. **Analysis dataset** Dimensions time, latitude/y, longitude/x [, ensemble_member]. When creating an analysis dataset from a forecast archive we take the shortest available lead time, flattening the init_time and lead_time dims into a single time dim.

**Vertical levels:** Single-level and surface variables live at the dataset root with the level encoded in the variable name (e.g. `temperature_2m`, `pressure_surface`). A variable available on a dense, comparable set of vertical levels does not have a level suffix and instead lives in a zarr group named after its vertical dimension — the group name and dimension name are the same (`pressure_level`, `model_level`; others may be added as needed), e.g. `pressure_level/temperature` is a variable with dimensions (time, latitude, longitude, pressure_level). Dimension coordinates shared with the root (time, lead time, latitude/longitude, ensemble_member, spatial_ref) are duplicated into each group so a group can be opened on its own. A group with no variables is omitted. A variable's group is a per-variable property (it sets the variable's zarr path and dims), not a job boundary. The materialized write path is not yet group-aware, so multi-group datasets are currently virtual only (a `DynamicalDataset` guard rejects a materialized dataset with any non-root variable); materialized multi-group support is planned.

**Variable naming:** A single-level or surface variable encodes its level in the name as `<var>_<level>` (e.g. `temperature_2m`); a variable carried on a vertical dimension is just `<var>` (the level lives in the dimension). Names also encode any aggregation; match an existing equivalent variable's name across datasets exactly (see Metadata conventions). Spell aggregation prefixes out in full — `maximum_`/`minimum_` (e.g. `maximum_wind_speed_10m`), not `max_`/`min_`. When a name spans a layer between two levels, order the two numbers to match the source GRIB level string (a `100-1000 mb` layer is `..._100_1000mb`; a `5000-2000 m` layer is `..._5000_2000m`).

**Dataset id and name:** A materialized dataset uses `dataset_id="<provider>-<model>-<variant>"` and `name="Provider Model variant"`. A virtual dataset carries a `-virtual` id suffix and a `, virtual` name suffix: `dataset_id="<provider>-<model>-<variant>-virtual"` and `name="Provider Model variant, virtual"`.

**Spatial dimensions:** If the source data uses a geographic projection we use dimensions latitude and longitude, else y and x are used for projected datasets.

## CLI commands

Run via `uv run main`.

### Global commands
- `uv run main --help` - Show all commands and registered datasets
- `uv run main initialize-new-integration <provider> <model> <variant>` - Scaffold new dataset
- `uv run main <dataset-id> update-template` - Regenerate `templates/latest.zarr`. Run this after any change to a `TemplateConfig` subclass's metadata.

## Parallelization model

Both backfills and operational updates distribute work across Kubernetes indexed jobs. Every worker independently computes the same ordered list of all jobs, then deterministically selects its subset. No coordinator or job queue is needed.

1. **Job generation**: `RegionJob.get_jobs()` creates all jobs by combining regions (shard slices along append dim) × variable groups (controlled by `max_vars_per_job`), with optional filters.

2. **Worker assignment**: `get_worker_jobs(all_jobs, worker_index, workers_total, worker_assignment=...)` distributes the append-dim-ordered jobs per the region job class's `worker_assignment` classvar — `"spread"` (bit-reversal permute then round-robin, the default) or `"contiguous"` (append-dim blocks, used by virtual region jobs to bound icechunk manifest rewrites).

3. **Coordination**: Workers coordinate via files in `_internal/{job_name}/` in object store. Worker 0 does setup, all workers process, the last worker (by index) finalizes.

4. **Reader safety**: Zarr v3 stores defer metadata writes until finalization. Icechunk commits are atomic.

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

Optimize the codebase you leave behind, not the size of your diff. Every change — code, architectural design, comments/docstrings, documentation — is judged by one outcome: is the **total codebase** afterward simpler, more maintainable, and easier to understand? Concretely: extend an existing utility or approach rather than introducing a parallel one; when a new approach is genuinely simpler and more general, migrate the old pattern into it rather than leaving both; a larger change that leaves one way of doing things beats a minimal change that leaves two. Documentation follows the same rule — each addition compounds, over many changes, toward either a codebase that explains itself or one buried in stale narratives. Always look for ways to simplify, and identify and suggest architectural improvements.

* Write code that explains itself rather than needs comments. See [Comments and docstrings](#comments-and-docstrings) below.
* Don't write error handing code unless I ask for it, nor smooth over exceptions/errors unless they are expected as part of control flow. In general, write code that will raise an exception early if something isn't expected. Enforce important expectations with asserts.
* Use Python 3.14+ features
* Follow ty type checking. If you need to add an ignore, ignore a specific check like `# ty: ignore[specific]`. Always annotate types on all function arguments and return types.
* Follow ruff format
* Test each module with pytest
* Log don't print: `from reformatters.common.logging import get_logger` and `log = get_logger(__name__)`
* Keep documentation up to date. After making a change that would update any .md file in the repo (CLAUDE.md, docs/*, etc.), always update relevant docs, regardless of if you have been explicitly asked.
* CLAUDE.md is an alias for AGENTS.md

### Comments and docstrings

The code is the sole source of truth: it cannot get out of date with itself; every comment and docstring can, and with no test to catch the rot, most eventually do. Each one is a standing liability charged to every future maintainer. The bar for writing one is not "is this true and helpful right now?" but: **will this still be true, and still be needed, by someone reading only the current code, long after this change is forgotten?**

Write for that reader: they have the current code and nothing else — no diff, no PR, no session transcript, no memory that anything was ever different. Whole categories of comment address someone watching the change happen and are noise to everyone after:

* How the code used to be, or that it changed ("now", "previously", "no longer").
* The debugging that led here. If code can't be made self-evident, state the failure mode it prevents as a timeless fact, not the story of finding it. (Exception: a test exists to pin a specific failure case, so describing that case is the test's contract.)
* Code, datasets, or examples consulted as reference during development.
* PRs and issues. Context that must outlive the change goes in an evergreen doc (docs/*.md); it almost never qualifies — when in doubt, drop it.

A docstring, when warranted, states the contract: what any caller may rely on without reading the body. Not internals — they change while the contract holds, and the body is right below (a non-obvious internal fact gets a comment at the line where it matters). Not callers — a function that names its callers points its dependencies backwards and narrows itself to today's usage.

Mechanics, when a comment does earn its place:
* Default is none. Comment only for a gotcha, out-of-the-ordinary behavior, or clarity lost to a necessary optimization. Don't remove existing comments.
* In non-test code, most comments are one line stating the non-obvious fact the reader needs — not reasoning, not defense of correctness, not operational instructions. For a genuinely complex topic, point to a doc instead (e.g. "..., see docs/parallel_processing.md.") — use sparingly; a doc pointer is still a comment and carries the same liability.
* An assert or validator with a clear message documents itself; don't add a comment restating it.

The spirit outranks the letter: a rare exception that truly serves the year-later reader is fine — use judgement. What is never fine is accumulation: individually-reasonable "helpful" notes compounding across changes into a codebase readers must wade through and learn to distrust. An agent's in-the-moment helpfulness is precisely this failure mode. Before ending a turn and before committing, reread every comment and docstring you added or edited, apply the bolded test to each sentence, and delete what fails — expect that to be most of it.
