This project contains code to reformat weather data into the zarr v3 file format.

## Approach overview

Datasets are created in 3 phases:
1. A template of the dataset, in the form of zarr metadata that is checked into the repo, is created with `uv run main <dataset_id> update-template`. This template (not in-code config) is loaded by steps 2 and 3 and drives processing and output in those steps. This approach of checking in the metadata allow us to review diffs if the structure or metadata of the dataset changes.
2. A zarr backfill is run. The backfill uses kubernetes indexed jobs to run work in parallel. When the user runs a `uv run main <dataset-id> backfill-kubernetes ...` command the metadata for the zarr is first written by the local process to the final zarr store, then a kubernetes index job is kicked off with each job index responsible for writing a portion of the zarr chunk data into the zarr archive.
3. Operational updates to the zarr are run using a kubernetes cronjob and validated by another kubernetes cronjob which runs after the update is expected to succeed. To ensure the archive is valid to readers throughout the update, the update writes data chunks for all data variables first, then updates the zarr metdata to reflect the larger dataset size. The operational update runs a single process to avoid interprocess communication while ensuring the metadata is updated last.

Common utilities and conventions seek to reduce the amount of unique code required for a single source of weather data/zarr dataset.

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
* Use a logger named `log`, use `print` only for temporary debugging
* Catch specific Exception subclasses unless there's a clear reason to catch a more general exception