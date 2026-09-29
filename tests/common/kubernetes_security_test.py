import json
import subprocess
from typing import Any
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from reformatters.__main__ import DYNAMICAL_DATASETS, OPERATIONAL_ARCHIVERS, app
from reformatters.common.kubernetes_security import verify_trigger_admission


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
    assert {
        "ecmwf-ifs-ens-46-day-daily-update",
        "ecmwf-ifs-ens-46-day-6-hourly-update",
    } <= expected
    result = CliRunner().invoke(
        app, ["render-admission-bundle", "--staging-target", "stage-update"]
    )
    assert result.exit_code == 0, result.exception
    resources: list[dict[str, Any]] = json.loads(result.stdout)["items"]
    names = {
        r["spec"]["paramRef"]["name"] for r in resources if "paramRef" in r["spec"]
    }
    assert names == expected | {"stage-update"}
