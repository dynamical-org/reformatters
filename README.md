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

To reformat a large archive we parallelize work across multiple cloud servers.

We use
* `docker` to package the code and dependencies
* `kubernetes` indexed jobs to run work in parallel

### Setup

1. Install `docker` and `kubectl`. Make sure `docker` can be found at `/usr/bin/docker` and `kubectl` at `/usr/bin/kubectl`.
1. Setup a docker image repository and export the `DOCKER_REPOSITORY` environment variable in your local shell. e.g. `export DOCKER_REPOSITORY=container.registry/<project-id>/reformatters/main`. Follow your registry's instructions to allow your docker to authenticate and push images to the registry.
1. Setup a kubernetes cluster and configure kubectl to point to your cluster. e.g. `aws eks update-kubeconfig --region <region> --name <cluster-name>`, `gcloud container clusters get-credentials <cluster-name> --region <region> --project <project>`, etc.
1. Create a kubectl secret containing a single json encoded value to be passed to fsspec `storage_options` or splatted as keyword arguments to an icechunk storage opener `kubectl create secret generic your-destination-storage-options-key --from-literal=contents='{"key": "...", "secret": "..."}'`. See `storage.py`.

1. As a cluster administrator, install the static trigger policies and give the deploy identity permission to create/get bindings and delegate the namespaced `trigger` verb:

   ```sh
   uv run main render-admission-bundle --namespace default > admission.json
   kubectl apply -f admission.json -f deploy/trigger-binding-deployer.yaml
   uv run main verify-admission admission.json --namespace default
   kubectl create clusterrolebinding reformatters-trigger-binding-deployer \
     --clusterrole=reformatters-trigger-binding-deployer --user=DEPLOY_IDENTITY
   kubectl create rolebinding reformatters-trigger-binding-deployer --namespace default \
     --role=reformatters-trigger-binding-deployer --user=DEPLOY_IDENTITY
   ```

   Deploy automatically creates bindings for every CronJob it deploys, probes them, then grants the `reformatters-update-trigger` account access. It never updates or deletes bindings or policies. The deploy identity also needs ordinary namespaced workload/RBAC management, CronJob `get`, and Job `create/get` permissions. No per-dataset admin step is needed. Do not grant wildcard verbs to the trigger account or its groups. Before pruning obsolete bindings, pause production and staging deploys and wait for in-flight deploys to finish, then revoke their targets from the trigger Role and wait for that change to reach every API server before deleting bindings; when replacing active policies, revoke the trigger RoleBinding until verification passes and the policies are active on all API servers.

1. Enable Restricted Pod Security warnings and audit before enforcement:

   ```sh
   kubectl label namespace default --overwrite \
     pod-security.kubernetes.io/warn=restricted \
     pod-security.kubernetes.io/warn-version=v1.36 \
     pod-security.kubernetes.io/audit=restricted \
     pod-security.kubernetes.io/audit-version=v1.36
   ```

   Check all workloads in the namespace, including those deployed elsewhere. Deploy the compliant templates, confirm replacement pods start and storage remains writable, and let Jobs using old templates finish. Old templates lacking the required security contexts will have replacement pods rejected. Once warnings are resolved, enforce:

   ```sh
   kubectl label namespace default --overwrite \
     pod-security.kubernetes.io/enforce=restricted \
     pod-security.kubernetes.io/enforce-version=v1.36
   ```
