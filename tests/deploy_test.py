import json
import subprocess
from collections.abc import Sequence
from datetime import timedelta
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import deploy, staging
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
    monkeypatch.setattr(deploy, "reconcile_heartbeats", mock_reconcile)
    monkeypatch.setattr(deploy, "write_heartbeat_secret", Mock())

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


def test_deploy_staging_skips_heartbeats(monkeypatch: pytest.MonkeyPatch) -> None:
    # Staging crons are renamed; no heartbeats are provisioned and no API key is needed.
    monkeypatch.setattr(subprocess, "run", Mock())
    monkeypatch.delenv("BETTERSTACK_API_KEY_RW", raising=False)
    mock_reconcile = Mock()
    monkeypatch.setattr(deploy, "reconcile_heartbeats", mock_reconcile)

    def transform(cronjob: CronJob) -> CronJob:
        return staging.rename_cronjob_for_staging(cronjob, "example-dataset-1", "0.0.1")

    deploy.deploy_operational_resources(
        [ExampleDataset1()],  # ty: ignore[invalid-argument-type]
        docker_image="test-image-tag",
        dataset_id_filter="example-dataset-1",
        cronjob_transform=transform,
    )
    mock_reconcile.assert_not_called()


def _cron(cls: type, name: str, schedule: str, deadline: timedelta) -> ReformatCronJob:
    return cls(
        name=name,
        schedule=schedule,
        pod_active_deadline=deadline,
        image="image:tag",
        dataset_id="example-dataset",
        cpu="1",
        memory="1G",
    )


def test_schedule_period() -> None:
    assert deploy.schedule_period("0 0 * * *") == timedelta(days=1)
    assert deploy.schedule_period("38 5,11,17,23 * * *") == timedelta(hours=6)


def test_heartbeat_specs() -> None:
    cron = _cron(
        ReformatCronJob, "example-dataset-update", "0 */6 * * *", timedelta(hours=2)
    )
    start, complete = deploy.heartbeat_specs(cron)

    assert start.key == "example-dataset_update_start"
    assert start.name == "reformatters example-dataset update start"
    assert start.period == timedelta(hours=6)
    assert start.grace == timedelta(minutes=10)

    assert complete.key == "example-dataset_update_complete"
    assert complete.grace == timedelta(minutes=10) + timedelta(hours=2)


def test_reconcile_heartbeats_create_and_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BETTERSTACK_API_KEY_RW", "test-token")

    # One heartbeat already exists with stale grace (forces a PATCH); others are created.
    existing_name = "reformatters example-dataset update start"
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "1",
                            "attributes": {
                                "name": existing_name,
                                "url": "https://hb.example/existing",
                                "period": 21600,
                                "grace": 1,
                            },
                        }
                    ],
                    "pagination": {"next": None},
                },
            )
        if request.method == "PATCH":
            return httpx.Response(200, json={"data": {"id": "1", "attributes": {}}})
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": "new",
                    "attributes": {"url": f"https://hb.example/{len(requests)}"},
                }
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        deploy,
        "_api_client",
        lambda token: httpx.Client(
            transport=transport, base_url="https://uptime.betterstack.com/api/v2"
        ),
    )

    cron_jobs = [
        _cron(
            ReformatCronJob, "example-dataset-update", "0 */6 * * *", timedelta(hours=1)
        ),
        _cron(
            ValidationCronJob,
            "example-dataset-validate",
            "0 */6 * * *",
            timedelta(hours=1),
        ),
    ]
    url_map = deploy.reconcile_heartbeats(cron_jobs)

    assert set(url_map) == {
        "example-dataset_update_start",
        "example-dataset_update_complete",
        "example-dataset_validate_start",
        "example-dataset_validate_complete",
    }
    assert ("PATCH", "/api/v2/heartbeats/1") in requests
    assert sum(1 for method, _ in requests if method == "POST") == 3


def test_provision_heartbeats_skips_non_eligible_crons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Archive and staging crons are filtered out, so no key is required and nothing is provisioned.
    monkeypatch.delenv("BETTERSTACK_API_KEY_RW", raising=False)
    mock_reconcile = Mock()
    monkeypatch.setattr(deploy, "reconcile_heartbeats", mock_reconcile)

    archive = CronJob(
        command=["archive-grib-files"],
        workers_total=1,
        parallelism=1,
        name="example-dataset-archive-grib-files",
        schedule="0 0 * * *",
        image="image:tag",
        dataset_id="example-dataset",
        cpu="1",
        memory="1G",
    )
    staging_update = _cron(
        ReformatCronJob,
        "stage-example-dataset-v2-update",
        "0 0 * * *",
        timedelta(hours=1),
    )

    deploy._provision_heartbeats([archive, staging_update])
    mock_reconcile.assert_not_called()
