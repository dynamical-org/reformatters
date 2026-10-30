import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import icechunk
import pytest
import s3fs
import yaml

from reformatters.common import dynamical_dataset, kubernetes, profiling, storage
from reformatters.common.config import Config, Env
from reformatters.common.kubernetes import Job
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.profiling import driver, worker
from reformatters.ecmwf.ifs_ens.profiling.config import (
    END,
    ROOT,
    START,
    ProfilingDataset,
    ProfilingStoreFactory,
    dataset,
)
from scripts.generate_manual_workflows import update_choice_options

RUN = "0123456789abcdef0123456789abcdef"
IMAGE = "example.invalid/reformatters:" + "a" * 40
REPO = Path(__file__).parents[3]


@pytest.fixture(autouse=True)
def isolated_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Unexpected credential, cluster, or remote storage access")

    monkeypatch.setattr(kubernetes, "load_secret", forbidden)
    monkeypatch.setattr(kubernetes.config, "load_kube_config", forbidden)
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", forbidden)
    monkeypatch.setattr(s3fs.S3FileSystem, "__init__", forbidden)
    monkeypatch.setattr(storage, "_get_icechunk_storage", forbidden)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    for name in (
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Config, "env", Env.prod)


def parameters() -> dict[str, Any]:
    return {
        "append_dim_end": END.to_pydatetime(),
        "filter_start": START.to_pydatetime(),
        "filter_end": END.to_pydatetime(),
        "jobs_per_pod": 1,
        "max_parallelism": 2,
        "docker_image": IMAGE,
        "profile": True,
    }


@pytest.mark.parametrize(("jobs_per_pod", "workers"), [(1, 38), (10, 4)])
def test_driver_uses_only_new_scratch_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, jobs_per_pod: int, workers: int
) -> None:
    seen: list[str] = []

    def local_storage(factory: ProfilingStoreFactory) -> icechunk.Storage:
        assert type(factory) is ProfilingStoreFactory
        factory.assert_destination()
        seen.append(factory.run_id)
        return icechunk.local_filesystem_storage(str(tmp_path / factory.run_id))

    def no_production(*args: object, **kwargs: object) -> None:
        pytest.fail("The profiling driver resolved production storage")

    monkeypatch.setattr(storage.StoreFactory, "primary_store", no_production)
    monkeypatch.setattr(storage.StoreFactory, "icechunk_repos", no_production)
    monkeypatch.setattr(storage.StoreFactory, "_coordination_fs", no_production)
    monkeypatch.setattr(ProfilingStoreFactory, "_storage", local_storage)
    submitted = Mock()
    monkeypatch.setattr(dynamical_dataset.subprocess, "run", submitted)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = parameters()
    args["jobs_per_pod"] = jobs_per_pod
    source = EcmwfIfsEnsForecast15Day025DegreeDataset(
        primary_storage_config=storage.StorageConfig(
            base_path="s3://example-invalid", format=storage.DatasetFormat.ICECHUNK
        )
    )
    source.backfill_kubernetes(**args)
    first = json.loads(submitted.call_args.kwargs["input"])
    source.backfill_kubernetes(**args)
    second = json.loads(submitted.call_args.kwargs["input"])
    assert len(set(seen)) == 2
    for manifest in [first, second]:
        run = manifest["metadata"]["labels"]["dynamical.org/profiling-run"]
        assert run in seen
        assert f"{ROOT}/{run}/data" in summary.read_text()
        assert manifest["spec"]["completions"] == workers
        assert manifest["spec"]["parallelism"] == 2
        assert manifest["metadata"]["name"].startswith("ifsens-profile-backfill-")
    assert IMAGE not in summary.read_text()
    assert "does not mean the Kubernetes Job is complete" in summary.read_text()


@pytest.mark.parametrize(
    "change",
    [
        {"append_dim_end": None},
        {"filter_start": None},
        {"filter_end": None},
        {"filter_contains": [START.to_pydatetime()]},
        {"filter_variable_names": ["temperature_2m"]},
        {"jobs_per_pod": 2},
        {"max_parallelism": 3},
        {"docker_image": "image:latest"},
        {"overwrite_chunks": True},
        {"overwrite_metadata": True},
        {"keep_branch": "candidate"},
    ],
)
def test_invalid_request_refuses_before_uuid_or_storage(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    def forbidden() -> None:
        pytest.fail("Allocated run id before request validation")

    monkeypatch.setattr(driver, "uuid4", forbidden)
    with pytest.raises(AssertionError):
        EcmwfIfsEnsForecast15Day025DegreeDataset(
            primary_storage_config=storage.StorageConfig(
                base_path="s3://example-invalid", format=storage.DatasetFormat.ICECHUNK
            )
        ).backfill_kubernetes(**(parameters() | change))


def test_profile_manifest_changes_only_explicit_fields() -> None:
    pilot = dataset(RUN)
    job = pilot._backfill_job(
        [
            "backfill",
            "2024-04-01T00:00:00",
            "--filter-start=2024-03-30T00:00:00",
            "--filter-end=2024-04-01T00:00:00",
        ],
        IMAGE,
        38,
        2,
        pilot._backfill_resource(),
    )
    base = Job(**{k: v for k, v in job.model_dump().items() if k != "run_id"})
    base.__dict__["job_name"] = job.job_name
    expected = base.as_kubernetes_object()
    actual = job.as_kubernetes_object()
    assert actual["spec"]["podReplacementPolicy"] == "Failed"
    assert actual["spec"]["activeDeadlineSeconds"] == 21600
    assert actual["spec"]["backoffLimitPerIndex"] == 0
    assert actual["spec"]["maxFailedIndexes"] == 0
    rules = actual["spec"]["podFailurePolicy"]["rules"]
    assert rules == [
        r
        for r in expected["spec"]["podFailurePolicy"]["rules"]
        if r["action"] != "Ignore"
    ]
    pod = actual["spec"]["template"]["spec"]
    assert pod["terminationGracePeriodSeconds"] == 60
    container = pod["containers"][0]
    assert IMAGE not in container["command"]
    assert container["command"][:4] == ["python", "-m", worker.MODULE, "run"]
    env = {v["name"]: v for v in container["env"]}
    assert "DYNAMICAL_SENTRY_DSN" not in env
    assert env["PROFILING_IMAGE"]["value"] == IMAGE
    assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
    normalized = copy.deepcopy(actual)
    normalized["metadata"].pop("labels")
    for key in ["activeDeadlineSeconds", "podReplacementPolicy"]:
        normalized["spec"].pop(key)
    for key in ["backoffLimitPerIndex", "maxFailedIndexes", "podFailurePolicy"]:
        normalized["spec"][key] = expected["spec"][key]
    normal_pod = normalized["spec"]["template"]["spec"]
    normal_pod["terminationGracePeriodSeconds"] = 30
    for key in ["command", "env"]:
        normal_pod["containers"][0][key] = expected["spec"]["template"]["spec"][
            "containers"
        ][0][key]
    assert normalized == expected


@pytest.mark.parametrize("workers", [4, 38])
def test_worker_assignment_covers_each_unit_once(
    monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    job = "ifsens-profile-backfill-test"
    monkeypatch.setenv("JOB_NAME", job)
    monkeypatch.setenv("WORKERS_TOTAL", str(workers))
    monkeypatch.setenv("PROFILING_IMAGE", IMAGE)
    assignments = []
    for index in range(workers):
        monkeypatch.setenv("WORKER_INDEX", str(index))
        pilot, metadata = worker._context(RUN, job, workers, IMAGE.rsplit(":", 1)[1])
        assert pilot.store_factory.primary_url().startswith(f"{ROOT}/{RUN}/data/")
        assert IMAGE not in json.dumps(metadata)
        assignments.extend(
            (tuple(a["region"]), a["variable"]) for a in metadata["assignment"]
        )
    assert len(assignments) == len(set(assignments)) == 38


def test_worker_runs_hooks_and_only_scratch_backfill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = "ifsens-profile-backfill-test"
    for name, value in {
        "JOB_NAME": job,
        "WORKERS_TOTAL": "38",
        "WORKER_INDEX": "0",
        "PROFILING_IMAGE": IMAGE,
    }.items():
        monkeypatch.setenv(name, value)
    bootstrap = Mock()
    backfill = Mock()
    monkeypatch.setattr(ProfilingStoreFactory, "bootstrap", bootstrap)
    monkeypatch.setattr(ProfilingDataset, "backfill", backfill)
    hooks = Mock(wraps=profiling.install_hooks)
    monkeypatch.setattr(profiling, "install_hooks", hooks)
    worker.run_worker(RUN, job, 38, IMAGE.rsplit(":", 1)[1], tmp_path)
    bootstrap.assert_called_once_with(0)
    backfill.assert_called_once()
    assert backfill.call_args.kwargs["workers_total"] == 38
    hooks.assert_called_once_with(tmp_path)


def test_default_cli_does_not_import_profiling() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import reformatters.__main__; import sys; assert not any('profiling' in name for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_supervisor_cli_passes_only_scratch_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = "ifsens-profile-backfill-test"
    pod_uid = "12345678-1234-1234-1234-123456789abc"
    sha = IMAGE.rsplit(":", 1)[1]
    for name, value in {
        "JOB_NAME": job,
        "WORKERS_TOTAL": "38",
        "WORKER_INDEX": "0",
        "PROFILING_IMAGE": IMAGE,
        "POD_UID": pod_uid,
    }.items():
        monkeypatch.setenv(name, value)
    fs = Mock()
    monkeypatch.setattr(ProfilingStoreFactory, "_coordination_fs", lambda self: fs)
    run_profiled = Mock(return_value=7)
    monkeypatch.setattr(profiling, "run_profiled", run_profiled)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "run",
            "--run-id",
            RUN,
            "--job-name",
            job,
            "--workers",
            "38",
            "--source-sha",
            sha,
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        worker.main()

    assert exit_info.value.code == 7
    run_profiled.assert_called_once()
    args = run_profiled.call_args.kwargs
    metadata = args["metadata"]
    attempt = metadata["attempt"]
    assert metadata["pod_uid"] == pod_uid
    assert metadata["run_id"] == RUN
    assert metadata["worker_index"] == 0
    assert metadata["source_sha"] == sha
    assert args["artifact_url"] == f"{ROOT}/{RUN}/profiles/{job}/{pod_uid}/{attempt}"
    assert args["fs"] is fs
    assert args["command"] == [
        sys.executable,
        "-m",
        worker.MODULE,
        "worker",
        "--run-id",
        RUN,
        "--job-name",
        job,
        "--workers",
        "38",
        "--source-sha",
        sha,
        "--output-dir",
        str(args["output_dir"]),
    ]
    assert IMAGE not in json.dumps(metadata)
    assert IMAGE not in " ".join(args["command"])


@pytest.mark.parametrize("enabled", ["", "false", "true"])
def test_workflow_profile_is_exact_boolean_and_generator_preserves_it(
    tmp_path: Path, enabled: str
) -> None:
    path = REPO / ".github/workflows/manual-backfill.yml"
    workflow = yaml.safe_load(path.read_text())
    assert workflow["on"]["workflow_dispatch"]["inputs"]["profile"]["default"] is False
    step = next(
        s
        for s in workflow["jobs"]["backfill"]["steps"]
        if s.get("name") == "Start backfill (SEE LOGS)"
    )
    script = step["run"].replace("${{ github.repository }}", "example/repo")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    fake_uv.chmod(0o755)
    capture = tmp_path / "argv"
    env = (
        os.environ
        | dict.fromkeys(step["env"], "")
        | {
            "PROFILE": enabled,
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "CAPTURE": str(capture),
            "DATASET_ID": "example",
            "OPERATION": "create-new-store",
            "DOCKER_IMAGE": IMAGE,
            "JOBS_PER_POD": "1",
            "MAX_PARALLELISM": "2",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        }
    )
    subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", script], env=env, check=True, capture_output=True
    )
    assert ("--profile" in capture.read_text().splitlines()) is (enabled == "true")
    copied = tmp_path / "workflow.yml"
    copied.write_text(path.read_text())
    update_choice_options(copied, "dataset_id", ["example"])
    updated = yaml.safe_load(copied.read_text())
    assert (
        updated["on"]["workflow_dispatch"]["inputs"]["profile"]
        == workflow["on"]["workflow_dispatch"]["inputs"]["profile"]
    )
    assert updated["jobs"] == workflow["jobs"]
