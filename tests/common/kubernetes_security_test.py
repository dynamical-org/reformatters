import json
import subprocess
from itertools import count
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from reformatters.__main__ import app
from reformatters.common import deploy, kubernetes_security
from reformatters.common.kubernetes_security import (
    create_trigger_bindings,
    trigger_admission_binding,
    trigger_admission_resources,
    trigger_role_targets,
    verify_trigger_admission,
)


def test_empty_targets_generate_bound_deny_all_guard() -> None:
    resources = trigger_admission_resources("default")
    assert [resource["kind"] for resource in resources] == [
        "ValidatingAdmissionPolicy",
        "ValidatingAdmissionPolicyBinding",
        "ValidatingAdmissionPolicy",
    ]
    assert resources[0]["metadata"]["name"] == (
        "default-reformatters-update-trigger-targets"
    )
    assert resources[1]["spec"]["policyName"] == resources[0]["metadata"]["name"]
    assert resources[1]["spec"]["validationActions"] == ["Deny"]
    assert (
        "authorizer.serviceAccount"
        in resources[0]["spec"]["validations"][0]["expression"]
    )
    assert (
        resources[2]["metadata"]["name"] == "default-reformatters-update-trigger-clone"
    )


@pytest.mark.parametrize(
    ("denial", "error"),
    [
        ("", "Admission allowed canary"),
        ("Forbidden: missing create permission", "Admission did not deny canary"),
    ],
)
def test_empty_targets_require_named_guard_denial(
    monkeypatch: pytest.MonkeyPatch, denial: str, error: str
) -> None:
    monkeypatch.setattr(kubernetes_security.time, "monotonic", count(0, 100).__next__)
    run = Mock(
        return_value=subprocess.CompletedProcess([], int(bool(denial)), stderr=denial)
    )
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError, match=error):
        verify_trigger_admission("default", [])
    run.assert_called_once()
    assert "--dry-run=server" in run.call_args.args[0]
    job = json.loads(run.call_args.kwargs["input"])
    assert job["metadata"]["labels"] == {}


def test_empty_targets_accept_named_guard_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [], 1, stderr="default-reformatters-update-trigger-targets"
        )
    )
    monkeypatch.setattr(subprocess, "run", run)
    verify_trigger_admission("default", [])
    run.assert_called_once()


@pytest.mark.parametrize("denial", ["", "Forbidden: missing create permission"])
def test_canary_requires_named_policy_denial(
    monkeypatch: pytest.MonkeyPatch, denial: str
) -> None:
    monkeypatch.setattr(kubernetes_security.time, "monotonic", count(0, 100).__next__)
    source = {
        "metadata": {"name": "example-update", "uid": "abc"},
        "spec": {
            "jobTemplate": {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{"name": "worker", "image": "approved"}]
                        }
                    }
                }
            }
        },
    }
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=json.dumps(source)),
            subprocess.CompletedProcess([], int(bool(denial)), stderr=denial),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError):
        verify_trigger_admission("default", ["example-update"])
    assert run.call_count == 2
    assert "--dry-run=server" in run.call_args.args[0]


def test_rendered_bundle_is_static() -> None:
    result = CliRunner().invoke(app, ["render-kubernetes-admission-bundle"])
    assert result.exit_code == 0, result.exception
    assert json.loads(result.stdout)["items"] == trigger_admission_resources("default")


def test_canary_retries_stale_parameter_denials_and_uses_unique_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "metadata": {"name": "example-update", "uid": "abc"},
        "spec": {
            "jobTemplate": {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{"name": "worker", "image": "approved"}]
                        }
                    }
                }
            }
        },
    }
    prefix = "default-reformatters-update-trigger"
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=json.dumps(source)),
            subprocess.CompletedProcess([], 1, stderr=f"{prefix}-targets"),
            subprocess.CompletedProcess([], 0, stderr=""),
            subprocess.CompletedProcess([], 1, stderr=f"{prefix}-clone"),
            subprocess.CompletedProcess([], 1, stderr=f"{prefix}-clone"),
            subprocess.CompletedProcess([], 0, stderr=""),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(kubernetes_security.time, "sleep", Mock())
    assert verify_trigger_admission("default", ["example-update"]) == {"example-update"}
    names = [
        json.loads(call.kwargs["input"])["metadata"]["name"]
        for call in run.call_args_list[1:]
    ]
    assert len(set(names)) == 5
    assert all("--dry-run=server" in call.args[0] for call in run.call_args_list[1:])


def test_operator_verifies_empty_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = "empty-triggers"
    items = trigger_admission_resources(namespace)
    bundle = tmp_path / "admission.json"
    bundle.write_text(json.dumps({"items": items}))

    def read_installed(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        item = next(
            item
            for item in items
            if item["metadata"]["name"] == command[3] and item["kind"] == command[2]
        )
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(item))

    run = Mock(side_effect=read_installed)
    monkeypatch.setattr(subprocess, "run", run)
    verify = Mock()
    monkeypatch.setattr(deploy, "verify_trigger_admission", verify)
    result = CliRunner().invoke(
        app, ["verify-kubernetes-admission", str(bundle), "--namespace", namespace]
    )
    assert result.exit_code == 0, result.exception
    assert run.call_count == len(items)
    verify.assert_called_once_with(namespace, [])

    result = CliRunner().invoke(app, ["verify-kubernetes-admission", str(bundle)])
    assert result.exit_code != 0
    assert "complete generated policy" in str(result.exception)
    assert run.call_count == len(items)


@pytest.mark.parametrize("already_exists", [False, True])
def test_binding_creation_is_idempotent_without_updates(
    monkeypatch: pytest.MonkeyPatch, already_exists: bool
) -> None:

    binding = trigger_admission_binding("default", "weather-update")
    stored = subprocess.CompletedProcess([], 0, stdout=json.dumps(binding))
    responses = (
        [subprocess.CompletedProcess([], 1, stderr="Error (AlreadyExists)")]
        if already_exists
        else []
    )
    run = Mock(side_effect=[*responses, stored])
    monkeypatch.setattr(subprocess, "run", run)
    create_trigger_bindings("default", ["weather-update"])
    assert [call.args[0][1] for call in run.call_args_list] == (
        ["create", "get"] if already_exists else ["create"]
    )


def test_conflicting_binding_fails_without_rewriting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 1, stderr="Error (AlreadyExists)"),
            subprocess.CompletedProcess(
                [], 0, stdout=json.dumps({"spec": {"validationActions": ["Warn"]}})
            ),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError, match="requires administrator review"):
        create_trigger_bindings("default", ["weather-update"])
    assert [call.args[0][1] for call in run.call_args_list] == ["create", "get"]


def test_binding_creation_permission_failure_is_not_already_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    run = Mock(return_value=subprocess.CompletedProcess([], 1, stderr="Forbidden"))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError, match="Forbidden"):
        create_trigger_bindings("default", ["weather-update"])
    run.assert_called_once()


@pytest.mark.parametrize("names", [None, []])
def test_unrestricted_trigger_role_is_rejected(
    monkeypatch: pytest.MonkeyPatch, names: list[str] | None
) -> None:

    rule: dict[str, Any] = {
        "apiGroups": ["batch"],
        "resources": ["cronjobs"],
        "verbs": ["trigger"],
    }
    if names is not None:
        rule["resourceNames"] = names
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [], 0, stdout=json.dumps({"rules": [rule]})
            )
        ),
    )
    with pytest.raises(AssertionError, match="must name their targets"):
        trigger_role_targets("default")


def test_trigger_role_targets_omits_deleted_cronjobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "rules": [
                            {
                                "apiGroups": ["batch"],
                                "resources": ["cronjobs"],
                                "verbs": ["get", "trigger"],
                                "resourceNames": ["live", "retired"],
                            }
                        ]
                    }
                ),
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "items": [
                            {"metadata": {"name": "live"}},
                            {"metadata": {"name": "foreign"}},
                        ]
                    }
                ),
            ),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    assert trigger_role_targets("default") == {"live"}
    assert run.call_count == 2
