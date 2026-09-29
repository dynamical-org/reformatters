import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from reformatters.__main__ import DYNAMICAL_DATASETS, OPERATIONAL_ARCHIVERS, app
from reformatters.common import deploy
from reformatters.common.kubernetes_security import (
    trigger_admission_resources,
    verify_trigger_admission,
)


def test_empty_targets_generate_bound_deny_all_guard() -> None:
    resources = trigger_admission_resources("default", [])
    assert [resource["kind"] for resource in resources] == [
        "ValidatingAdmissionPolicy",
        "ValidatingAdmissionPolicyBinding",
        "ValidatingAdmissionPolicy",
    ]
    assert resources[0]["metadata"]["name"] == (
        "default-reformat-update-trigger-targets"
    )
    assert resources[1]["spec"]["policyName"] == resources[0]["metadata"]["name"]
    assert resources[1]["spec"]["validationActions"] == ["Deny"]
    assert " in []" in resources[0]["spec"]["validations"][0]["expression"]
    assert resources[2]["metadata"]["name"] == "default-reformat-update-trigger-clone"


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
    run = Mock(
        return_value=subprocess.CompletedProcess([], int(bool(denial)), stderr=denial)
    )
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError, match=error):
        verify_trigger_admission("default", [], require_params=False)
    run.assert_called_once()
    assert "--dry-run=server" in run.call_args.args[0]
    job = json.loads(run.call_args.kwargs["input"])
    assert job["metadata"]["labels"]["dynamical.org/cronjob-name"] == "unapproved"


def test_empty_targets_accept_named_guard_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [], 1, stderr="default-reformat-update-trigger-targets"
        )
    )
    monkeypatch.setattr(subprocess, "run", run)
    verify_trigger_admission("default", [], require_params=True)
    run.assert_called_once()


@pytest.mark.parametrize("denial", ["", "Forbidden: missing create permission"])
def test_canary_requires_named_policy_denial(
    monkeypatch: pytest.MonkeyPatch, denial: str
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
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=json.dumps(source)),
            subprocess.CompletedProcess([], int(bool(denial)), stderr=denial),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError):
        verify_trigger_admission("default", [source], require_params=True)
    assert run.call_count == 2
    assert "--dry-run=server" in run.call_args.args[0]


def test_rendered_bundle_targets_registered_triggerable_cronjobs() -> None:
    expected = set()
    for resource in [*DYNAMICAL_DATASETS, *OPERATIONAL_ARCHIVERS]:
        try:
            cronjobs = resource.operational_kubernetes_resources("unused")
        except NotImplementedError:
            continue
        expected.update(cj.name for cj in cronjobs if cj.triggerable)
    assert expected == set()
    result = CliRunner().invoke(
        app, ["render-admission-bundle", "--staging-target", "stage-update"]
    )
    assert result.exit_code == 0, result.exception
    resources: list[dict[str, Any]] = json.loads(result.stdout)["items"]
    names = {
        r["spec"]["paramRef"]["name"] for r in resources if "paramRef" in r["spec"]
    }
    assert names == expected | {"stage-update"}


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
    prefix = "default-reformat-update-trigger"
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=json.dumps(source)),
            subprocess.CompletedProcess([], 1, stderr=f"{prefix}-targets"),
            subprocess.CompletedProcess([], 1, stderr=f"{prefix}-clone"),
            subprocess.CompletedProcess([], 1, stderr=f"{prefix}-clone"),
            subprocess.CompletedProcess([], 0, stderr=""),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    verify_trigger_admission("default", [source], require_params=True)
    names = [
        json.loads(call.kwargs["input"])["metadata"]["name"]
        for call in run.call_args_list[1:]
    ]
    assert len(set(names)) == 4
    assert all("--dry-run=server" in call.args[0] for call in run.call_args_list[1:])


def test_operator_verifies_every_bundle_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["render-admission-bundle", "--staging-target", "stage-update"]
    )
    assert result.exit_code == 0, result.exception
    bundle = tmp_path / "admission.json"
    bundle.write_text(result.stdout)
    items = json.loads(result.stdout)["items"]

    def read_installed(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        item = next(
            item
            for item in items
            if item["metadata"]["name"] == command[3] and item["kind"] == command[2]
        )
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(item))

    monkeypatch.setattr(subprocess, "run", read_installed)
    verify = Mock()
    monkeypatch.setattr(deploy, "verify_trigger_admission", verify)
    result = runner.invoke(app, ["verify-admission", str(bundle)])
    assert result.exit_code == 0, result.exception
    assert "stage-update" in {t["metadata"]["name"] for t in verify.call_args.args[1]}
    assert verify.call_args.kwargs == {"require_params": False}

    contents = json.loads(bundle.read_text())
    contents["items"] = [
        item
        for item in contents["items"]
        if item["spec"].get("paramRef", {}).get("name") != "stage-update"
    ]
    bundle.write_text(json.dumps(contents))
    result = runner.invoke(app, ["verify-admission", str(bundle)])
    assert result.exit_code != 0
    assert "complete generated policy" in str(result.exception)
    verify.assert_called_once()


def test_operator_verifies_empty_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = "empty-triggers"
    items = trigger_admission_resources(namespace, [])
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
        app, ["verify-admission", str(bundle), "--namespace", namespace]
    )
    assert result.exit_code == 0, result.exception
    assert run.call_count == len(items)
    verify.assert_called_once_with(namespace, [], require_params=False)

    result = CliRunner().invoke(app, ["verify-admission", str(bundle)])
    assert result.exit_code != 0
    assert "complete generated policy" in str(result.exception)
    assert run.call_count == len(items)
