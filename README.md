# dynamical.org reformatters
Reformat weather datasets into zarr.

## Local development

We use 
* `uv` to manage dependencies and python environments
* `ruff` for linting and formatting
* `mypy` for type checking
* `pytest` for testing
* `pre-commit` to automatically lint and format as you git commit

### Setup
1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
1. Run `uv pip install -e .` to install the project in editable mode
1. Run `uv run pre-commit install` to setup the git hooks
1. If you use VSCode, you may want to install the extensions (ruff, mypy) it will recommend when you open this folder

### Running locally

* `uv run main --help`
* `uv run main noaa-gefs-forecast update-template`
* `uv run main noaa-gefs-forecast reformat-local 2024-01-02T00:00`

### Development commands
* Add dependency: `uv add <package> [--dev]`. Use `--dev` to add a development only dependency.
* Lint: `uv run ruff check`
* Type check: `uv run mypy`
* Format: `uv run ruff format`
* Test: `uv run pytest`

## Deploying to the cloud

To reformat a large archive we parallelize work across multiple cloud servers.

We use
* `docker` to package the code and dependencies
* `kubernetes` indexed jobs to run work in parallel

### Setup

1. Install `docker` and `kubectl`. Make sure `docker` can be found at /usr/bin/docker and `kubectl` at /usr/bin/kubectl.
1. Setup a docker image repository and export the DOCKER_REPOSITORY environment variable in your local shell. eg. `export DOCKER_REPOSITORY=us-central1-docker.pkg.dev/<project-id>/reformatters/main`
1. Setup a kubernetes cluster and configure kubectl to point to your cluster. eg `gcloud container clusters get-credentials <cluster-name> --region <region> --project <project>`
1. Create a kubectl secret containing your Source Coop S3 credentials `kubectl create secret generic source-coop-key --from-literal='AWS_ACCESS_KEY_ID=XXX' --from-literal='AWS_SECRET_ACCESS_KEY=XXX'`


### Development commands
1. `uv run main noaa-gefs-forecast reformat-kubernetes <INIT_TIME_END> [--jobs-per-pod <int>] [--max-parallelism <int>]`