import json
import subprocess
from collections.abc import Iterable
from datetime import timedelta
from typing import Any
from unittest.mock import Mock

import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import deploy
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import Job, ReformatCronJob, ValidationCronJob


class ExampleDatasetInDevelopment:
    dataset_id: str = "example-dataset-in-dev"

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[Job]:
        # This should not be deployed, nor cause issues with other deploys
        raise NotImplementedError("this dataset is in development")


class ExampleDataset1:
    dataset_id: str = "example-dataset-1"

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[Job]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-operational-update",
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
            name=f"{self.dataset_id}-validation",
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


def test_deploy_operational_updates(monkeypatch: pytest.MonkeyPatch) -> None:
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
    )  # type: ignore[assignment]

    deploy.deploy_operational_updates(test_datasets, docker_image="test-image-tag")

    assert mock_run.call_count == 1
    args, kwargs = mock_run.call_args
    assert args[0] == ["/usr/bin/kubectl", "apply", "-f", "-"]

    resources = json.loads(kwargs["input"])
    assert resources["apiVersion"] == "v1"
    assert resources["kind"] == "List"

    # Dataset 1
    assert resources["items"][0]["kind"] == "CronJob"
    assert (
        resources["items"][0]["metadata"]["name"]
        == "example-dataset-1-operational-update"
    )
    container_spec = resources["items"][0]["spec"]["jobTemplate"]["spec"]["template"][
        "spec"
    ]["containers"][0]
    assert container_spec["resources"] == {"requests": {"cpu": "14", "memory": "30G"}}
    assert container_spec["image"] == "test-image-tag"

    # Dataset 2
    assert resources["items"][2]["kind"] == "CronJob"
    assert (
        resources["items"][2]["metadata"]["name"]
        == "example-dataset-2-operational-update"
    )
