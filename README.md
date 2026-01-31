# dynamical.org reformatters
Reformat weather datasets into zarr.

Browse the datasets produced by this repo at https://dynamical.org/catalog/.

* See [AGENTS.md](AGENTS.md) for an overview of the approach and this repository.
* [Integrate a new dataset](docs/dataset_integration_guide.md) to be reformatted.
* [Add a new variable](docs/add_new_variable.md) to an existing dataset.

## Local development

We use 
* `uv` to manage dependencies and python environments
* `ruff` for linting and formatting
* `mypy` for type checking
* `pytest` for testing
* `prek` to automatically lint and format as you git commit

### Setup
1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
1. Run `uv run prek install` to setup the git hooks
1. If you use VSCode, you may want to install the extensions (ruff, mypy) it will recommend when you open this folder

### Running locally

* `uv run main --help` - list all datasets
* `uv run main <DATASET_ID> update-template`
* `uv run main <DATASET_ID> backfill-local <INIT_TIME_END>`

### Development commands
* Add dependency: `uv add <package> [--dev]`. Use `--dev` to add a development only dependency.
* Lint: `uv run ruff check [--fix]`
* Type check: `uv run mypy`
* Format: `uv run ruff format`
* Tests: 
   * Run tests in parallel on all available cores: `uv run pytest`
   * Run tests serially: `uv run pytest -n 0`

## Deploying to the cloud

To reformat a large archive we parallelize work across multiple cloud servers.

We use
* `docker` to package the code and dependencies
* `kubernetes` indexed jobs to run work in parallel

### Setup

1. Install `docker` and `kubectl`. Make sure `docker` can be found at `/usr/bin/docker` and `kubectl` at `/usr/bin/kubectl`.
1. Setup a docker image repository and export the `DOCKER_REPOSITORY` environment variable in your local shell. e.g. `export DOCKER_REPOSITORY=container.registry/<project-id>/reformatters/main`. Follow your registry's instructions to allow your docker to authenticate and push images to the registry.
1. Setup a kubernetes cluster and configure kubectl to point to your cluster. e.g. `aws eks update-kubeconfig --region <region> --name <cluster-name>`, `gcloud container clusters get-credentials <cluster-name> --region <region> --project <project>`, etc.
1. Create a kubectl secret containing a single json encoded value to be passed to fsspec `storage_options` or splatted as keyword arguments to an icechunk storage opener `kubectl create secret generic your-destination-storage-options-key --from-literal=contents='{"key": "...", "secret": "..."}'`. See `storage.py`.