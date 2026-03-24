from typing import Any

import pytest
from typer.testing import CliRunner

from reformatters.__main__ import DYNAMICAL_DATASETS, app
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import ReformatCronJob, ValidationCronJob

runner = CliRunner()


DATASET_IDS = [d.dataset_id for d in DYNAMICAL_DATASETS]


# --- Registry tests ---


def test_no_duplicate_dataset_ids() -> None:
    assert len(DATASET_IDS) == len(set(DATASET_IDS)), (
        f"Duplicate dataset IDs: {[d for d in DATASET_IDS if DATASET_IDS.count(d) > 1]}"
    )


def test_all_datasets_have_unique_versions() -> None:
    """Each dataset_id + version combination should be unique to avoid store path collisions."""
    id_version_pairs = [
        (d.dataset_id, d.template_config.version) for d in DYNAMICAL_DATASETS
    ]
    assert len(id_version_pairs) == len(set(id_version_pairs))


# --- CLI wiring tests ---


@pytest.mark.parametrize("dataset_id", DATASET_IDS)
def test_cli_help_works_for_dataset(dataset_id: str) -> None:
    """Verify the dataset subcommand is registered and --help doesn't crash."""
    result = runner.invoke(app, [dataset_id, "--help"])
    assert result.exit_code == 0, (
        f"{dataset_id} --help failed: {result.output}\n{result.exception}"
    )


@pytest.mark.parametrize("dataset_id", DATASET_IDS)
def test_cli_has_update_command(dataset_id: str) -> None:
    result = runner.invoke(app, [dataset_id, "update", "--help"])
    assert result.exit_code == 0, (
        f"{dataset_id} update --help failed: {result.output}\n{result.exception}"
    )


@pytest.mark.parametrize("dataset_id", DATASET_IDS)
def test_cli_has_validate_command(dataset_id: str) -> None:
    result = runner.invoke(app, [dataset_id, "validate", "--help"])
    assert result.exit_code == 0, (
        f"{dataset_id} validate --help failed: {result.output}\n{result.exception}"
    )


@pytest.mark.parametrize("dataset_id", DATASET_IDS)
def test_cli_has_update_template_command(dataset_id: str) -> None:
    result = runner.invoke(app, [dataset_id, "update-template", "--help"])
    assert result.exit_code == 0, (
        f"{dataset_id} update-template --help failed: {result.output}\n{result.exception}"
    )


@pytest.mark.parametrize("dataset_id", DATASET_IDS)
def test_cli_has_backfill_kubernetes_command(dataset_id: str) -> None:
    result = runner.invoke(app, [dataset_id, "backfill-kubernetes", "--help"])
    assert result.exit_code == 0, (
        f"{dataset_id} backfill-kubernetes --help failed: {result.output}\n{result.exception}"
    )


@pytest.mark.parametrize("dataset_id", DATASET_IDS)
def test_cli_has_process_backfill_region_jobs_command(dataset_id: str) -> None:
    result = runner.invoke(app, [dataset_id, "process-backfill-region-jobs", "--help"])
    assert result.exit_code == 0, (
        f"{dataset_id} process-backfill-region-jobs --help failed: {result.output}\n{result.exception}"
    )


def test_cli_has_deploy_command() -> None:
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0, f"deploy --help failed: {result.output}"


def test_cli_has_deploy_staging_command() -> None:
    result = runner.invoke(app, ["deploy-staging", "--help"])
    assert result.exit_code == 0, f"deploy-staging --help failed: {result.output}"


def test_cli_has_cleanup_staging_command() -> None:
    result = runner.invoke(app, ["cleanup-staging", "--help"])
    assert result.exit_code == 0, f"cleanup-staging --help failed: {result.output}"


def test_cli_has_initialize_new_integration_command() -> None:
    result = runner.invoke(app, ["initialize-new-integration", "--help"])
    assert result.exit_code == 0, (
        f"initialize-new-integration --help failed: {result.output}"
    )


# --- Operational configuration tests ---


@pytest.mark.parametrize(
    "dataset",
    DYNAMICAL_DATASETS,
    ids=DATASET_IDS,
)
def test_operational_kubernetes_resources_are_valid(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """Every dataset must define operational_kubernetes_resources that returns valid CronJobs."""
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image"))

    assert len(cron_jobs) >= 2, (
        f"{dataset.dataset_id}: expected at least an update and validate cronjob"
    )

    reformat_jobs = [c for c in cron_jobs if isinstance(c, ReformatCronJob)]
    validation_jobs = [c for c in cron_jobs if isinstance(c, ValidationCronJob)]

    assert len(reformat_jobs) >= 1, f"{dataset.dataset_id}: missing ReformatCronJob"
    assert len(validation_jobs) >= 1, f"{dataset.dataset_id}: missing ValidationCronJob"


@pytest.mark.parametrize(
    "dataset",
    DYNAMICAL_DATASETS,
    ids=DATASET_IDS,
)
def test_cronjob_commands_match_cli_commands(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """Every CronJob command must correspond to a real CLI command for that dataset."""
    cli = dataset.get_cli()
    # Typer stores registered commands in cli.registered_commands and groups in cli.registered_groups
    # Get the actual command names from the typer app
    registered_command_names = set()
    for command in cli.registered_commands:
        name = command.name or command.callback.__name__  # type: ignore[union-attr]
        # typer converts underscores to hyphens
        registered_command_names.add(name.replace("_", "-"))

    cron_jobs = list(dataset.operational_kubernetes_resources("test-image"))
    for cron_job in cron_jobs:
        # The first element of the command list is the CLI command name
        command_name = cron_job.command[0]
        assert command_name in registered_command_names, (
            f"{dataset.dataset_id}: CronJob '{cron_job.name}' uses command "
            f"'{command_name}' which is not a registered CLI command. "
            f"Available commands: {registered_command_names}"
        )


@pytest.mark.parametrize(
    "dataset",
    DYNAMICAL_DATASETS,
    ids=DATASET_IDS,
)
def test_cronjob_names_are_consistent(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """CronJob names should contain the dataset ID for traceability."""
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image"))
    for cron_job in cron_jobs:
        assert dataset.dataset_id in cron_job.name, (
            f"CronJob name '{cron_job.name}' doesn't contain dataset_id '{dataset.dataset_id}'"
        )


@pytest.mark.parametrize(
    "dataset",
    DYNAMICAL_DATASETS,
    ids=DATASET_IDS,
)
def test_cronjob_kubernetes_manifests_are_valid(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """Every CronJob must produce a valid kubernetes manifest."""
    cron_jobs = list(dataset.operational_kubernetes_resources("test-image"))
    for cron_job in cron_jobs:
        manifest = cron_job.as_kubernetes_object()
        assert manifest["kind"] == "CronJob"
        assert manifest["metadata"]["name"] == cron_job.name

        job_spec = manifest["spec"]["jobTemplate"]["spec"]
        container = job_spec["template"]["spec"]["containers"][0]

        # The command should start with python, the entrypoint, and the dataset_id
        assert container["command"][:3] == [
            "python",
            "src/reformatters/__main__.py",
            dataset.dataset_id,
        ]

        # Verify env vars that cronjobs depend on
        env_names = {e["name"] for e in container["env"]}
        assert "DYNAMICAL_ENV" in env_names
        assert "JOB_NAME" in env_names
        assert "CRON_JOB_NAME" in env_names


@pytest.mark.parametrize(
    "dataset",
    DYNAMICAL_DATASETS,
    ids=DATASET_IDS,
)
def test_validators_are_defined(dataset: DynamicalDataset[Any, Any]) -> None:
    """Every dataset must define validators."""
    validators = list(dataset.validators())
    assert len(validators) >= 1, f"{dataset.dataset_id}: no validators defined"


@pytest.mark.parametrize(
    "dataset",
    DYNAMICAL_DATASETS,
    ids=DATASET_IDS,
)
def test_dataset_id_matches_template_config(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    assert dataset.dataset_id == dataset.template_config.dataset_id


@pytest.mark.parametrize(
    "dataset",
    DYNAMICAL_DATASETS,
    ids=DATASET_IDS,
)
def test_store_factory_k8s_secret_names_not_empty_for_datasets_with_replicas(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """Datasets with replica stores need k8s secrets configured."""
    if dataset.replica_storage_configs:
        secret_names = dataset.store_factory.k8s_secret_names()
        assert len(secret_names) >= 1, (
            f"{dataset.dataset_id}: has replicas but no k8s secret names"
        )
