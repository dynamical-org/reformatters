import pytest

from scripts.retained_ifs_repair import Manifest
from scripts.retained_ifs_repair_job import job_definition


def definition(**overrides: str | bool | None) -> dict:
    options = {
        "name": "retained-repair-pilot",
        "namespace": "default",
        "image": "registry/image@sha256:" + "a" * 64,
        "manifest": Manifest.model_construct(base_snapshot="baseline"),
        "manifest_sha256": "b" * 64,
        "input_claim": "reviewed-inputs",
        "output_claim": "repair-evidence",
        "memory": "35Gi",
        "shared_memory": "19Gi",
        "scratch": "10Gi",
        "storage_secret": None,
        "commit": False,
        "capacity": "on-demand",
    }
    options.update(overrides)
    return job_definition(**options)


def test_pilot_is_one_nonretrying_worker_and_readonly_inputs() -> None:
    job = definition()
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["completions"] == job["spec"]["parallelism"] == 1
    pod = job["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert pod["automountServiceAccountToken"] is False
    worker = pod["containers"][0]
    assert "--commit-to-new-branch" not in worker["command"]
    assert "--auth" not in worker["command"]
    assert worker["resources"]["limits"]["cpu"] == "4"
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    assert worker["volumeMounts"][0]["readOnly"] is True
    assert {item["name"]: item["value"] for item in worker["env"]}[
        "NUMBA_NUM_THREADS"
    ] == "1"


def test_commit_has_explicit_secret_and_flag() -> None:
    storage_name = "storage-options"
    pod = definition(commit=True, storage_secret=storage_name)["spec"]["template"][
        "spec"
    ]
    command = pod["containers"][0]["command"]
    assert command[-1] == "--commit-to-new-branch"
    assert command[command.index("--auth") + 1] == "secret"
    assert pod["volumes"][-1]["secret"]["items"] == [
        {"key": "contents", "path": "storage-options.json"}
    ]
    assert pod["containers"][0]["volumeMounts"][-1]["readOnly"] is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"image": "registry/image:main"}, "digest"),
        ({"commit": True}, "secret"),
        ({"storage_secret": "write-secret"}, "secret"),
        (
            {"commit": True, "storage_secret": "storage", "capacity": "spot"},
            "on-demand",
        ),
        ({"input_claim": "repair-evidence"}, "distinct"),
        ({"name": "../bad"}, "name"),
        ({"manifest_sha256": "unreviewed"}, "hash"),
    ],
)
def test_invalid_job_inputs(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        definition(**changes)
