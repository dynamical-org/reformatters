import json
import subprocess
import pytest
from reformatters.common import deploy

class DummyJob:
    def __init__(self):
        self.obj = {"metadata": {"name": "dummy-job"}, "spec": {}}
    def as_kubernetes_object(self):
        return self.obj

class DummyDataset:
    def operational_kubernetes_resources(self, image_tag: str):
        assert image_tag == "test-image-tag"
        return [DummyJob()]

def test_deploy_operational_updates(monkeypatch):
    # Prevent legacy resources from polluting test
    monkeypatch.setattr(deploy, "LEGACY_OPERATIONAL_RESOURCE_FNS", ())
    # Capture subprocess.run calls
    calls = []
    def fake_run(cmd, input, text, check):
        calls.append({"cmd": cmd, "input": input, "text": text, "check": check})
    monkeypatch.setattr(subprocess, "run", fake_run)
    # Invoke deploy with our dummy dataset and image tag
    deploy.deploy_operational_updates([DummyDataset()], docker_image="test-image-tag")
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
