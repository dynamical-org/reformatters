# dynamical.org reformatters
Reformat weather datasets into zarr.

## Local development

We use 
* `uv` to manage dependencies and python environments
* `ruff` for linting and formatting
* `mypy` for type checking
* `pre-commit` to automatically lint and format as you git commit (type checking on commit is TODO)

### Setup
1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
1. Run `uv run pre-commit install` to setup the git hooks
1. If you use VSCode, you may want to install the extensions (ruff, mypy) it will recommend when you open this folder

### Running locally

* `uv run main.py --help`
* `uv run main.py noaa-gefs-forecast update-template`
* `uv run main.py noaa-gefs-forecast reformat-local 2024-01-02T00:00`

### Development commands
* Add dependency: `uv add <package> [--dev]`. Use `--dev` to add a development only dependency.
* Lint: `uv run ruff check`
* Type check: `uv run mypy`
* Format: `uv run ruff format`


## Deploying to the cloud

To reformat a large archive we parallelize work across multiple cloud servers.

We use
* `docker` to containerize the code
* `kubernetes` indexed jobs to run work in parallel

### Setup

1. Install `docker` and `kubectl`. Make sure `docker` can be found at /usr/bin/docker and `kubectl` at /usr/bin/kubectl.
1. Setup a docker image repository and export the DOCKER_REPOSITORY environment variable in your local shell. eg. `export DOCKER_REPOSITORY=us-central1-docker.pkg.dev/<project-id>/reformatters/main`
1. Setup a kubernetes cluster and configure kubectl to point to your cluster. eg `gcloud container clusters get-credentials <cluster-name> --region <region> --project <project>`
1. Create a kubectl secret containing your Source Coop S3 credentials `kubectl create secret generic source-coop-key --from-literal='AWS_ACCESS_KEY_ID=XXX' --from-literal='AWS_SECRET_ACCESS_KEY=XXX'`

### WIP Setup instructions for accessing the new AWS cluster

1. Install the [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
1. Get invited to create an AWS Account and an IAM User by someone in the org; accept the email invites to set up your AWS account & IAM user.
1. Navigate to the IAM portal (should auto-redirect upon accepting the IAM invite). Underneath your account in the dropdown, click on "access keys".
1. Run `aws configure sso` in your terminal. Copy-paste the SSO Start URL & SSO Region from the IAM access keys page, and then follow the SSO url that is output in your terminal to auth your account in the browser. 

Scratch notes on how to invite people, since it's surprisingly convoluted:
- invite people to a new AWS account [via this page](https://us-east-1.console.aws.amazon.com/organizations/v2/home/accounts).
- IAM Users can be created [here](https://us-east-1.console.aws.amazon.com/singlesignon/home?region=us-east-1#!/instances/7223601ab3bfa0c6/users)
- Then from within the IAM Portal, [assign the account to the IAM User](https://us-east-1.console.aws.amazon.com/singlesignon/organization/home?region=us-east-1#/instances/7223601ab3bfa0c6/accounts/add-users). Official docs are [here](https://docs.aws.amazon.com/singlesignon/latest/userguide/assignusers.html). At the end of that flow, you can choose to give that account/user the Admin Access role, or a less permissive role of your choosing. 
- **note**: when scaling this to more users, we probably want to use IAM roles instead of individual IAM users. [See suggestion in the docs here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html).

### Development commands
1. `uv run main.py noaa-gefs-forecast reformat-kubernetes <INIT_TIME_END> [--jobs-per-pod <int>] [--max-parallelism <int>]`