import json
import subprocess
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

import pytest

from reformatters.common import deploy
from reformatters.common.kubernetes import Job, ReformatCronJob, ValidationCronJob


class ExampleDataset1:
    dataset_id: str = "example-dataset-1"

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[Job]:
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-operational-update",
            schedule="",
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
            schedule="",
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
    # Prevent legacy resources from polluting test
    monkeypatch.setattr(deploy, "LEGACY_OPERATIONAL_RESOURCE_FNS", ())
    # Use unitest.Mock or similar instead to capture calls AI!
    # Capture subprocess.run calls
    calls: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], input: str, text: bool, check: bool) -> None:
        calls.append({"cmd": cmd, "input": input, "text": text, "check": check})

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Invoke deploy with our dummy dataset and image tag
    deploy.deploy_operational_updates(
        [ExampleDataset1(), ExampleDataset2()],  # type: ignore[list-item]
        docker_image="test-image-tag",
    )
    # Verify subprocess.run was called exactly once
    assert len(calls) == 1
    call = calls[0]
    assert call["cmd"] == ["/usr/bin/kubectl", "apply", "-f", "-"]
    # Parse JSON payload
    payload = json.loads(call["input"])
    assert payload["apiVersion"] == "v1"
    assert payload["kind"] == "List"
    # Verify that our DummyJob's object appears in items
    assert payload["items"] == [DummyJob().as_kubernetes_object()]
