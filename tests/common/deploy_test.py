import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import timedelta
from importlib.metadata import distribution
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import deploy, monitoring
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import (
    SERVICE_ACCOUNT,
    CronJob,
    ReformatCronJob,
    ValidationCronJob,
)


class ExampleDatasetInDevelopment:
    dataset_id: str = "example-dataset-in-dev"

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # This should not be deployed, nor cause issues with other deploys
        raise NotImplementedError("this dataset is in development")


class ExampleDataset1:
    dataset_id: str = "example-dataset-1"

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="0 0 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="14",
            memory="30G",
            shared_memory="12G",
            ephemeral_storage="30G",
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="0 0 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
        )

        return [operational_update_cron_job, validation_cron_job]


class ExampleDataset2(ExampleDataset1):
    dataset_id: str = "example-dataset-2"


@pytest.fixture(autouse=True)
def admission_gate(monkeypatch: pytest.MonkeyPatch) -> Mock:
    gate = Mock()
    monkeypatch.setattr(deploy, "verify_trigger_admission", gate)
    return gate


def test_deploy_operational_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock()
    monkeypatch.setattr(subprocess, "run", mock_run)

    example_datasets = [
        ExampleDatasetInDevelopment(),
        ExampleDataset1(),
        ExampleDataset2(),
    ]

    # Also add in the real datasets to test they don't cause errors.
    # They are last in the list so their results don't impact the indexes we verify below.
    test_datasets: list[DynamicalDataset[Any, Any]] = example_datasets + list(
        DYNAMICAL_DATASETS
    )  # ty: ignore[invalid-assignment]

    deploy.deploy_operational_resources(test_datasets, docker_image="test-image-tag")

    assert mock_run.call_count == 2
    cronjob_apply, rbac_apply = mock_run.call_args_list
    args, kwargs = cronjob_apply
    assert args[0] == ["/usr/bin/kubectl", "apply", "--namespace", "default", "-f", "-"]

    resources = json.loads(kwargs["input"])
    service_account = resources["items"].pop(0)
    assert service_account["kind"] == "ServiceAccount"
    assert resources["apiVersion"] == "v1"
    assert resources["kind"] == "List"

    # Dataset 1
    assert resources["items"][0]["kind"] == "CronJob"
    assert resources["items"][0]["metadata"]["name"] == "example-dataset-1-update"
    assert resources["items"][1]["metadata"]["name"] == "example-dataset-1-validate"
    container_spec = resources["items"][0]["spec"]["jobTemplate"]["spec"]["template"][
        "spec"
    ]["containers"][0]
    assert container_spec["resources"] == {"requests": {"cpu": "14", "memory": "30G"}}
    assert container_spec["image"] == "test-image-tag"

    # Dataset 2
    assert resources["items"][2]["kind"] == "CronJob"
    assert resources["items"][2]["metadata"]["name"] == "example-dataset-2-update"
    assert resources["items"][3]["metadata"]["name"] == "example-dataset-2-validate"

    rbac_resources = json.loads(rbac_apply.kwargs["input"])["items"]
    rbac = {item["kind"]: item for item in rbac_resources}
    assert set(rbac) == {"Role", "RoleBinding"}
    assert [item["kind"] for item in rbac_resources] == [
        "Role",
        "RoleBinding",
    ]
    assert all(item["metadata"]["name"] == SERVICE_ACCOUNT for item in rbac.values())
    assert rbac["Role"]["rules"] == [
        {"apiGroups": ["batch"], "resources": ["cronjobs"], "verbs": ["get"]},
        {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create", "get"]},
    ]
    assert rbac["RoleBinding"]["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": SERVICE_ACCOUNT,
    }
    assert rbac["RoleBinding"]["subjects"] == [
        {"kind": "ServiceAccount", "name": SERVICE_ACCOUNT}
    ]


def test_deploy_operational_resources_dataset_id_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_run = Mock()
    monkeypatch.setattr(subprocess, "run", mock_run)

    test_datasets: list[DynamicalDataset[Any, Any]] = [
        ExampleDataset1(),
        ExampleDataset2(),
    ]  # ty: ignore[invalid-assignment]

    deploy.deploy_operational_resources(
        test_datasets,
        docker_image="test-image-tag",
        dataset_id_filter="example-dataset-2",
    )

    resources = json.loads(mock_run.call_args_list[0].kwargs["input"])
    names = [
        item["metadata"]["name"]
        for item in resources["items"]
        if item["kind"] == "CronJob"
    ]
    assert names == ["example-dataset-2-update", "example-dataset-2-validate"]
    rbac_resources = json.loads(mock_run.call_args_list[1].kwargs["input"])["items"]
    role = next(item for item in rbac_resources if item["kind"] == "Role")
    assert role["rules"][0]["resources"] == ["cronjobs"]
    assert "resourceNames" not in role["rules"][0]


@pytest.mark.parametrize("staging", [False, True])
def test_partial_deploy_only_verifies_selected_targets(
    monkeypatch: pytest.MonkeyPatch, admission_gate: Mock, staging: bool
) -> None:
    monkeypatch.setattr(subprocess, "run", Mock())
    deploy.deploy_operational_resources(
        DYNAMICAL_DATASETS,
        docker_image="test-image-tag",
        dataset_id_filter="ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree",
        cronjob_transform=(
            lambda cj: cj.model_copy(update={"name": "staging-" + cj.command[0]})
        )
        if staging
        else None,
    )
    assert admission_gate.call_count == 2
    for call in admission_gate.call_args_list:
        assert [t["metadata"]["name"] for t in call.args[1]] == [
            "staging-update" if staging else "ecmwf-ifs-ens-46-day-daily-update"
        ]


@pytest.mark.parametrize("failed_gate", [0, 1])
def test_deploy_stops_before_grant_when_admission_fails(
    monkeypatch: pytest.MonkeyPatch,
    admission_gate: Mock,
    failed_gate: int,
) -> None:
    mock_run = Mock()
    monkeypatch.setattr(subprocess, "run", mock_run)
    admission_gate.side_effect = (
        [AssertionError("policy missing")]
        if failed_gate == 0
        else [None, AssertionError("clone denied")]
    )
    with pytest.raises(AssertionError):
        deploy.deploy_operational_resources(
            DYNAMICAL_DATASETS, docker_image="test-image-tag"
        )
    assert mock_run.call_count == failed_gate
    for call in mock_run.call_args_list:
        assert all(
            item["kind"] not in {"Role", "RoleBinding"}
            for item in json.loads(call.kwargs["input"])["items"]
        )


def test_deploy_stops_before_grant_when_cronjob_apply_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_run = Mock(side_effect=subprocess.CalledProcessError(1, "kubectl"))
    monkeypatch.setattr(subprocess, "run", mock_run)
    with pytest.raises(subprocess.CalledProcessError):
        deploy.deploy_operational_resources(
            DYNAMICAL_DATASETS, docker_image="test-image-tag"
        )
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["check"] is True


def test_registered_dataset_schedules_are_parseable() -> None:
    # Operational virtual updates derive their poll deadline from the schedule, so an
    # unparseable one would surface as a failed update rather than a failed test.
    for dataset in DYNAMICAL_DATASETS:
        try:
            cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")
        except NotImplementedError:
            continue
        for cron_job in cron_jobs:
            assert cron_job.previous_fire_time(pd.Timestamp("2026-08-02T12:34")) < (
                pd.Timestamp("2026-08-02T12:34")
            )


def test_console_entrypoint_dispatch_installs_sigterm_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = next(
        entrypoint
        for entrypoint in distribution("reformatters").entry_points
        if entrypoint.group == "console_scripts" and entrypoint.name == "main"
    )
    assert entrypoint.value == "reformatters.__main__:app"

    install_sigterm_logger = Mock()
    monkeypatch.setattr(monitoring, "install_sigterm_logger", install_sigterm_logger)
    log_peak_memory = Mock()
    monkeypatch.setattr(monitoring, "log_peak_memory", log_peak_memory)
    result = CliRunner().invoke(
        entrypoint.load(), ["noaa-hrrr-analysis-virtual", "dataset-urls"]
    )

    assert result.exit_code == 0, result.exception
    install_sigterm_logger.assert_called_once_with()
    log_peak_memory.assert_called_once_with()


def test_direct_file_dispatch_installs_sigterm_logger() -> None:
    code = """
import runpy
import sys
from unittest.mock import Mock

from reformatters.common import monitoring

install_sigterm_logger = Mock()
monitoring.install_sigterm_logger = install_sigterm_logger
log_peak_memory = Mock()
monitoring.log_peak_memory = log_peak_memory
sys.argv = [
    "src/reformatters/__main__.py",
    "noaa-hrrr-analysis-virtual",
    "dataset-urls",
]
try:
    runpy.run_path("src/reformatters/__main__.py", run_name="__main__")
except SystemExit as error:
    assert error.code == 0
install_sigterm_logger.assert_called_once_with()
log_peak_memory.assert_called_once_with()
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class TestDeployCommandsRegistered:
    def test_deploy_commands_in_cli(self) -> None:
        from reformatters.__main__ import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "deploy " in result.output or "deploy\n" in result.output
        assert "deploy-staging" in result.output
        assert "cleanup-staging" in result.output
