# dynamical.org reformatters

Reformat weather datasets into zarr.

Browse the datasets produced by this repo at https://dynamical.org/catalog/.

* See [AGENTS.md](AGENTS.md) for an overview of the approach and this repository.
* [Develop a new dataset](docs/dataset_development_guide.md) end to end, or dive into just the [implementation](docs/implementation_guide.md).
* [Add a new variable](docs/add_new_variable.md) to an existing dataset.

[![DOI](https://zenodo.org/badge/859043226.svg)](https://doi.org/10.5281/zenodo.18777399)

## Local development

We use
* `uv` to manage dependencies and python environments
* `ruff` for linting and formatting
* `ty` for type checking
* `pytest` for testing
* `prek` to automatically lint and format as you git commit

### Setup
1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
1. Run `uv run prek install` to setup the git hooks
1. If you use VSCode, you may want to install the extensions (ruff) it will recommend when you open this folder

### Running locally

* `uv run main --help` - list all datasets
* `uv run main <DATASET_ID> update-template`
* `uv run main <DATASET_ID> backfill-local <APPEND_DIM_END>`

### Development commands
* Add dependency: `uv add <package> [--dev]`. Use `--dev` to add a development only dependency.
* Lint: `uv run ruff check [--fix]`
* Type check: `uv run ty check`
* Format: `uv run ruff format`
* Tests: 
   * Run tests in parallel on all available cores: `uv run pytest`
   * Run tests serially: `uv run pytest -n 0`

## Deploying to the cloud

To reformat a large archive we parallelize work across multiple cloud servers. See [docs/cluster_setup.md](docs/cluster_setup.md).
