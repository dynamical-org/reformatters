import json
import subprocess
from collections.abc import Sequence
from datetime import timedelta
from typing import Any
from unittest.mock import Mock

import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import betterstack, deploy
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob


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


def test_deploy_operational_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock()
    monkeypatch.setattr(subprocess, "run", mock_run)

    monkeypatch.setenv("BETTERSTACK_API_KEY_RW", "test-token")
    mock_reconcile = Mock(return_value={})
    monkeypatch.setattr(betterstack, "reconcile_heartbeats", mock_reconcile)
    monkeypatch.setattr(betterstack, "write_heartbeat_secret", Mock())

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

    assert mock_run.call_count == 1
    args, kwargs = mock_run.call_args
    assert args[0] == ["/usr/bin/kubectl", "apply", "-f", "-"]

    resources = json.loads(kwargs["input"])
    assert resources["apiVersion"] == "v1"
    assert resources["kind"] == "List"

    # Dataset 1
    assert resources["items"][0]["kind"] == "CronJob"
    assert resources["items"][0]["metadata"]["name"] == "example-dataset-1-update"
    container_spec = resources["items"][0]["spec"]["jobTemplate"]["spec"]["template"][
        "spec"
    ]["containers"][0]
    assert container_spec["resources"] == {"requests": {"cpu": "14", "memory": "30G"}}
    assert container_spec["image"] == "test-image-tag"

    # Dataset 2
    assert resources["items"][2]["kind"] == "CronJob"
    assert resources["items"][2]["metadata"]["name"] == "example-dataset-2-update"

    # Heartbeats were provisioned for the deployed cron jobs before kubectl apply.
    assert mock_reconcile.call_count == 1
    provisioned_names = {cj.name for cj in mock_reconcile.call_args.args[0]}
    assert {
        "example-dataset-1-update",
        "example-dataset-1-validate",
    } <= provisioned_names


def test_deploy_requires_betterstack_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", Mock())
    monkeypatch.delenv("BETTERSTACK_API_KEY_RW", raising=False)

    with pytest.raises(RuntimeError, match="BETTERSTACK_API_KEY_RW is required"):
        deploy.deploy_operational_resources(
            [ExampleDataset1()],  # ty: ignore[invalid-argument-type]
            docker_image="test-image-tag",
        )
