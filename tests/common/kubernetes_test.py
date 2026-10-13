import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from pydantic import ValidationError

from reformatters.__main__ import DYNAMICAL_DATASETS, OPERATIONAL_ARCHIVERS
from reformatters.common.config import Config, Env
from reformatters.common.kubernetes import (
    SERVICE_ACCOUNT,
    VALIDATION_FAILURE_EXIT_CODE,
    CronJob,
    Job,
    ReformatCronJob,
    _load_secret_from_kubernetes_api,
    create_job_from_cronjob,
    has_running_cronjob_job,
    load_secret,
    retry_job_name,
)


def test_as_kubernetes_object_comprehensive() -> None:
    """Test that as_kubernetes_object returns the expected structure and configuration."""
    job = Job(
        command=["backfill-kubernetes", "2025-01-01T00:00:00", "1", "1"],
        image="weather-app:v1.0",
        dataset_id="weather_data",
        cpu="500m",
        memory="1Gi",
        workers_total=4,
        parallelism=2,
        secret_names=["aws-creds", "db-creds"],
    )

    k8s_obj: dict[str, Any] = job.as_kubernetes_object()

    # Test top-level structure
    assert k8s_obj["apiVersion"] == "batch/v1"
    assert k8s_obj["kind"] == "Job"

    # Test metadata
    assert "name" in k8s_obj["metadata"]
    assert k8s_obj["metadata"]["name"].startswith("weather-data-backfill-kubernetes-")

    # Test complete spec
    expected_spec = {
        "backoffLimitPerIndex": 5,
        "completionMode": "Indexed",
        "completions": 4,
        "maxFailedIndexes": 4,
        "parallelism": 2,
        "podFailurePolicy": {
            "rules": [
                {
                    "action": "Ignore",
                    "onPodConditions": [{"type": "DisruptionTarget", "status": "True"}],
                },
                {
                    "action": "FailJob",
                    "onPodConditions": [{"type": "ConfigIssue", "status": "True"}],
                },
                {
                    "action": "FailJob",
                    "onExitCodes": {
                        "containerName": "worker",
                        "operator": "In",
                        "values": [VALIDATION_FAILURE_EXIT_CODE],
                    },
                },
            ]
        },
        "template": {
            "metadata": {"annotations": {"karpenter.sh/do-not-disrupt": "true"}},
            "spec": {
                "containers": [
                    {
                        "command": [
                            "python",
                            "src/reformatters/__main__.py",
                            "weather_data",
                            "backfill-kubernetes",
                            "2025-01-01T00:00:00",
                            "1",
                            "1",
                        ],
                        "env": [
                            {"name": "DYNAMICAL_ENV", "value": "prod"},
                            {
                                "name": "DYNAMICAL_SENTRY_DSN",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "key": "DYNAMICAL_SENTRY_DSN",
                                        "name": "sentry",
                                    }
                                },
                            },
                            {
                                "name": "JOB_NAME",
                                "valueFrom": {
                                    "fieldRef": {
                                        "fieldPath": "metadata.labels['job-name']"
                                    }
                                },
                            },
                            {
                                "name": "POD_NAME",
                                "valueFrom": {
                                    "fieldRef": {"fieldPath": "metadata.name"}
                                },
                            },
                            {
                                "name": "POD_NAMESPACE",
                                "valueFrom": {
                                    "fieldRef": {"fieldPath": "metadata.namespace"}
                                },
                            },
                            {
                                "name": "WORKER_INDEX",
                                "valueFrom": {
                                    "fieldRef": {
                                        "fieldPath": "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                                    }
                                },
                            },
                            {
                                "name": "WORKERS_TOTAL",
                                "value": "4",
                            },
                        ],
                        "image": "weather-app:v1.0",
                        "name": "worker",
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]},
                        },
                        "resources": {
                            "requests": {
                                "cpu": "500m",
                                "memory": "1Gi",
                            }
                        },
                        "volumeMounts": [
                            {"mountPath": "/app/data", "name": "ephemeral-vol"},
                            {
                                "name": "aws-creds",
                                "mountPath": "/secrets/aws-creds.json",
                                "subPath": "contents",
                                "readOnly": True,
                            },
                            {
                                "name": "db-creds",
                                "mountPath": "/secrets/db-creds.json",
                                "subPath": "contents",
                                "readOnly": True,
                            },
                        ],
                    }
                ],
                "nodeSelector": {
                    "eks.amazonaws.com/compute-type": "auto",
                    "karpenter.sh/capacity-type": "spot",
                },
                "restartPolicy": "Never",
                "securityContext": {
                    "fsGroup": 999,
                    "runAsNonRoot": True,
                    "runAsUser": 999,
                    "runAsGroup": 999,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "terminationGracePeriodSeconds": 30,
                "activeDeadlineSeconds": 21600,  # default 6 hours
                "volumes": [
                    {
                        "name": "ephemeral-vol",
                        "ephemeral": {
                            "volumeClaimTemplate": {
                                "metadata": {"labels": {"type": "ephemeral"}},
                                "spec": {
                                    "accessModes": ["ReadWriteOnce"],
                                    "resources": {
                                        "requests": {"storage": "10G"}  # default value
                                    },
                                },
                            }
                        },
                    },
                    {"name": "aws-creds", "secret": {"secretName": "aws-creds"}},
                    {"name": "db-creds", "secret": {"secretName": "db-creds"}},
                ],
            },
        },
        "ttlSecondsAfterFinished": 86400,  # default 24 hours
    }

    assert k8s_obj["spec"] == expected_spec


def _assert_restricted_security(pod_spec: dict[str, Any]) -> None:
    assert pod_spec["securityContext"] == {
        "fsGroup": 999,
        "runAsNonRoot": True,
        "runAsUser": 999,
        "runAsGroup": 999,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    assert len(pod_spec["containers"]) == 1
    assert pod_spec["containers"][0]["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }


def test_registry_workload_templates_use_restricted_security() -> None:
    for resource in (*DYNAMICAL_DATASETS, *OPERATIONAL_ARCHIVERS):
        workloads = list(resource.operational_kubernetes_resources("test-image"))
        assert workloads, resource.dataset_id
        for workload in workloads:
            manifest = workload.as_kubernetes_object()
            assert manifest["kind"] == "CronJob", workload.name
            pod_spec = manifest["spec"]["jobTemplate"]["spec"]["template"]["spec"]
            _assert_restricted_security(pod_spec)


def test_backfill_job_template_uses_restricted_security() -> None:
    job = Job(
        command=["backfill-kubernetes"],
        image="img:v1",
        dataset_id="weather-data",
        cpu="1",
        memory="1Gi",
        shared_memory="512Mi",
        workers_total=2,
        parallelism=2,
        secret_names=["source-creds"],
    )

    pod_spec = job.as_kubernetes_object()["spec"]["template"]["spec"]
    _assert_restricted_security(pod_spec)
    assert {mount["name"] for mount in pod_spec["containers"][0]["volumeMounts"]} == {
        "ephemeral-vol",
        "shared-memory-dir",
        "source-creds",
    }


def _cron_job_with_name(name: str, cron_job_class: type[CronJob] = CronJob) -> CronJob:
    return cron_job_class(
        name=name,
        schedule="0 * * * *",
        command=[name.rsplit("-", maxsplit=1)[-1]],
        image="img:v1",
        dataset_id="archive",
        cpu="1",
        memory="1Gi",
        workers_total=1,
        parallelism=1,
    )


@pytest.mark.parametrize(
    ("cron_job_class", "suffix"),
    [
        (CronJob, "-update"),
        (ReformatCronJob, "-update"),
    ],
)
def test_cron_job_name_respects_kubernetes_length_limit(
    cron_job_class: type[CronJob], suffix: str
) -> None:
    """Every operational cron job class must enforce the limit, not just the base class.

    A subclass that redeclares `name` without nesting `CronJobName` silently drops
    the constraint.
    """
    _cron_job_with_name("a" * (52 - len(suffix)) + suffix, cron_job_class)
    with pytest.raises(ValidationError, match="at most 52 characters"):
        _cron_job_with_name("a" * (53 - len(suffix)) + suffix, cron_job_class)


def test_do_not_disrupt_annotation_on_every_job() -> None:
    reformat = ReformatCronJob(
        name="weather-data-update",
        schedule="0 * * * *",
        image="img:v1",
        dataset_id="weather_data",
        cpu="1",
        memory="1Gi",
    )
    backfill = Job(
        command=["backfill-kubernetes"],
        image="img:v1",
        dataset_id="weather_data",
        cpu="1",
        memory="1Gi",
        workers_total=1,
        parallelism=1,
    )

    def pod_template(obj: dict[str, Any], *, cron: bool) -> dict[str, Any]:
        spec = obj["spec"]["jobTemplate"]["spec"] if cron else obj["spec"]
        return spec["template"]

    def annotations(obj: dict[str, Any], *, cron: bool) -> dict[str, str]:
        return pod_template(obj, cron=cron)["metadata"]["annotations"]

    do_not_disrupt = {"karpenter.sh/do-not-disrupt": "true"}
    assert annotations(reformat.as_kubernetes_object(), cron=True) == do_not_disrupt
    # A backfill worker carries many region jobs, so it loses the most to an eviction.
    assert annotations(backfill.as_kubernetes_object(), cron=False) == do_not_disrupt


def test_kubernetes_job_name() -> None:
    """Test ensure that the job name is consistent across invocations"""
    job = Job(
        command=["backfill-kubernetes", "2025-01-01T00:00:00", "1", "1"],
        image="weather-app:v1.0",
        dataset_id="weather_data",
        cpu="500m",
        memory="1Gi",
        workers_total=4,
        parallelism=2,
        secret_names=["aws-creds", "db-creds"],
    )

    k8s_obj: dict[str, Any] = job.as_kubernetes_object()
    assert job.job_name == k8s_obj["metadata"]["name"]
    assert job.job_name == job.job_name  # quick explicit check that result is cached


def test_as_kubernetes_object_with_custom_values() -> None:
    """Test as_kubernetes_object with custom resource values."""
    job = Job(
        command=["validate"],
        image="validator:latest",
        dataset_id="custom_dataset",
        cpu="2000m",
        memory="4Gi",
        shared_memory="512Mi",
        ephemeral_storage="50G",
        workers_total=8,
        parallelism=4,
        ttl=timedelta(hours=2),
        pod_active_deadline=timedelta(hours=3),
    )

    k8s_obj: dict[str, Any] = job.as_kubernetes_object()

    # Test custom TTL
    assert k8s_obj["spec"]["ttlSecondsAfterFinished"] == 7200  # 2 hours

    # Test pod active deadline
    pod_spec: dict[str, Any] = k8s_obj["spec"]["template"]["spec"]
    assert pod_spec["activeDeadlineSeconds"] == 10800  # 3 hours

    # Read volumes and volume mounts into dicts for easier access
    volumes: dict[str, dict[str, Any]] = {
        vol["name"]: vol for vol in pod_spec["volumes"]
    }
    assert len(pod_spec["containers"]) == 1  # assumed in [0] access just below
    volume_mounts: dict[str, dict[str, Any]] = {
        mount["name"]: mount for mount in pod_spec["containers"][0]["volumeMounts"]
    }

    # Test shared memory volume
    shared_mem_vol: dict[str, Any] = volumes["shared-memory-dir"]
    assert shared_mem_vol["emptyDir"]["sizeLimit"] == "512Mi"
    assert shared_mem_vol["emptyDir"]["medium"] == "Memory"
    assert volume_mounts["shared-memory-dir"]["mountPath"] == "/dev/shm"  # noqa: S108

    # Test ephemeral storage
    ephemeral_vol: dict[str, Any] = volumes["ephemeral-vol"]
    storage_request: str = ephemeral_vol["ephemeral"]["volumeClaimTemplate"]["spec"][
        "resources"
    ]["requests"]["storage"]
    assert storage_request == "50G"


@pytest.mark.parametrize(
    ("workers_total", "expected_max_failed_indexes"),
    [
        # Small worker count: min(100, max(min(5, 3), 3 // 8)) = min(100, max(3, 0)) = 3
        (3, 3),
        # Edge case: min(5, workers_total) wins: min(100, max(min(5, 5), 5 // 8)) = min(100, max(5, 0)) = 5
        (5, 5),
        # Medium count: workers_total // 8 wins: min(100, max(min(5, 40), 40 // 8)) = min(100, max(5, 5)) = 5
        (40, 5),
        # Larger count: workers_total // 8 wins: min(100, max(min(5, 64), 64 // 8)) = min(100, max(5, 8)) = 8
        (64, 8),
        # Very large: hits 100 limit: min(100, max(min(5, 1000), 1000 // 8)) = min(100, max(5, 125)) = 100
        (1000, 100),
    ],
)
def test_max_failed_indexes_calculation(
    workers_total: int, expected_max_failed_indexes: int
) -> None:
    """Test that maxFailedIndexes is calculated correctly for different worker counts."""
    job = Job(
        command=["test"],
        image="test:latest",
        dataset_id="test",
        cpu="100m",
        memory="128Mi",
        workers_total=workers_total,
        parallelism=1,
    )

    k8s_obj = job.as_kubernetes_object()
    assert k8s_obj["spec"]["maxFailedIndexes"] == expected_max_failed_indexes


def test_load_secret_returns_empty_dict_in_non_prod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that load_secret returns empty dict when not in prod environment."""
    monkeypatch.setattr(Config, "env", Env.dev)
    result = load_secret("test-secret")
    assert result == {}

    monkeypatch.setattr(Config, "env", Env.test)
    result = load_secret("test-secret")
    assert result == {}


def test_load_secret_from_mounted_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that load_secret loads from mounted file in prod when file exists."""
    monkeypatch.setattr(Config, "env", Env.prod)
    monkeypatch.setattr(
        "reformatters.common.kubernetes._SECRET_MOUNT_PATH", str(tmp_path)
    )

    secret_data = {"key1": "value1", "key2": 42, "key3": True}
    secret_file = tmp_path / "test-secret.json"
    secret_file.write_text(json.dumps(secret_data))

    result = load_secret("test-secret")
    assert result == secret_data


def test_load_secret_raises_when_file_missing_in_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that load_secret raises FileNotFoundError when file missing and JOB_NAME is set."""
    monkeypatch.setattr(Config, "env", Env.prod)
    monkeypatch.setattr(
        "reformatters.common.kubernetes._SECRET_MOUNT_PATH", str(tmp_path)
    )
    monkeypatch.setenv("JOB_NAME", "test-job")

    with pytest.raises(
        FileNotFoundError, match=r"Secret file .* not found in production job"
    ):
        load_secret("missing-secret")


def test_load_secret_from_kubernetes_api_when_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that load_secret falls back to kubernetes API when running locally (no JOB_NAME)."""
    monkeypatch.setattr(Config, "env", Env.prod)
    monkeypatch.setattr(
        "reformatters.common.kubernetes._SECRET_MOUNT_PATH", str(tmp_path)
    )
    monkeypatch.delenv("JOB_NAME", raising=False)

    secret_data = {"api_key": "secret123", "count": 99}

    with patch(
        "reformatters.common.kubernetes._load_secret_from_kubernetes_api"
    ) as mock_load:
        mock_load.return_value = secret_data
        result = load_secret("test-secret")

    assert result == secret_data
    mock_load.assert_called_once_with("test-secret")


def test_load_secret_from_kubernetes_api() -> None:
    """Test _load_secret_from_kubernetes_api loads and decodes secret from kubernetes API."""
    secret_data = {"username": "admin", "password": "secret", "port": 5432}
    secret_json = json.dumps(secret_data)
    encoded_secret = base64.b64encode(secret_json.encode("utf-8")).decode("utf-8")

    mock_secret = Mock()
    mock_secret.data = {"contents": encoded_secret}

    mock_v1 = MagicMock()
    mock_v1.read_namespaced_secret.return_value = mock_secret

    with (
        patch("reformatters.common.kubernetes.config.load_kube_config"),
        patch("reformatters.common.kubernetes.client.CoreV1Api", return_value=mock_v1),
    ):
        result = _load_secret_from_kubernetes_api("db-credentials")

    assert result == secret_data
    mock_v1.read_namespaced_secret.assert_called_once_with("db-credentials", "default")


def _cron_volume_names(cron: CronJob) -> set[str]:
    spec = cron.as_kubernetes_object()["spec"]["jobTemplate"]["spec"]["template"][
        "spec"
    ]
    return {volume["name"] for volume in spec["volumes"]}


def test_update_cron_mounts_dataset_secrets() -> None:
    cron = ReformatCronJob(
        name="weather-data-update",
        schedule="0 * * * *",
        image="img",
        dataset_id="weather-data",
        cpu="1",
        memory="1G",
        secret_names=["source-creds"],
    )
    assert "source-creds" in _cron_volume_names(cron)


def _cron_job(schedule: str) -> ReformatCronJob:
    return ReformatCronJob(
        name="weather-data-update",
        schedule=schedule,
        image="img",
        dataset_id="weather-data",
        cpu="1",
        memory="1G",
    )


@pytest.mark.parametrize(
    ("schedule", "now", "expected"),
    [
        ("50 * * * *", "2026-08-02T23:55", "2026-08-02T23:50"),
        ("50 * * * *", "2026-08-02T23:50", "2026-08-02T23:50"),
        # Before this hour's fire, so the previous hour's.
        ("50 * * * *", "2026-08-02T23:12", "2026-08-02T22:50"),
        ("53 1,7,13,19 * * *", "2026-08-02T03:00", "2026-08-02T01:53"),
        # Walks back into the previous day.
        ("53 1,7,13,19 * * *", "2026-08-02T01:00", "2026-08-01T19:53"),
        ("0 20 * * *", "2026-08-02T19:59", "2026-08-01T20:00"),
        ("13 */6 * * *", "2026-08-02T07:00", "2026-08-02T06:13"),
        ("17 1-23/3 * * *", "2026-08-02T01:00", "2026-08-01T22:17"),
        ("0 23 */3 * *", "2026-08-02T12:34", "2026-08-01T23:00"),
        ("0 23 */3 * *", "2026-09-01T22:00", "2026-08-31T23:00"),
        ("0 23 */3 * *", "2026-10-01T22:00", "2026-09-28T23:00"),
    ],
)
def test_previous_fire_time(schedule: str, now: str, expected: str) -> None:
    assert _cron_job(schedule).previous_fire_time(pd.Timestamp(now)) == pd.Timestamp(
        expected
    )


@pytest.mark.parametrize(
    "schedule",
    [
        "*/5 * * * *",  # a stepped minute field selects more than one fire an hour
        "0 0 * * 1",  # day of week
        "0 0 1 * *",  # day of month
        "5 1,2/2 * * *",  # a step within a list
        "0 24 * * *",
        "60 0 * * *",
    ],
)
def test_previous_fire_time_rejects_unsupported_schedule(schedule: str) -> None:
    with pytest.raises(AssertionError):
        _cron_job(schedule).previous_fire_time(pd.Timestamp("2026-08-02T12:00"))


@pytest.mark.parametrize(
    ("schedule", "now", "expected"),
    [
        ("0 * * * *", "2026-08-02T12:00", "2026-08-02T13:00"),
        ("53 1,7,13,19 * * *", "2026-08-02T19:53", "2026-08-03T01:53"),
        ("0 23 */3 * *", "2026-09-28T23:00", "2026-10-01T23:00"),
    ],
)
def test_next_fire_time(schedule: str, now: str, expected: str) -> None:
    assert _cron_job(schedule).next_fire_time(pd.Timestamp(now)) == pd.Timestamp(
        expected
    )


def test_update_cronjob_has_service_account_and_validation_failure_policy() -> None:
    cron = _cron_job("0 * * * *")
    spec = cron.as_kubernetes_object()["spec"]["jobTemplate"]["spec"]
    assert spec["template"]["spec"]["serviceAccountName"] == SERVICE_ACCOUNT
    assert {
        "action": "FailJob",
        "onExitCodes": {
            "containerName": "worker",
            "operator": "In",
            "values": [VALIDATION_FAILURE_EXIT_CODE],
        },
    } in spec["podFailurePolicy"]["rules"]
    assert (
        "serviceAccountName"
        not in Job(
            command=["backfill-kubernetes"],
            image="img",
            dataset_id="weather-data",
            cpu="1",
            memory="1G",
            workers_total=1,
            parallelism=1,
        ).as_kubernetes_object()["spec"]["template"]["spec"]
    )


@pytest.fixture
def running_cronjob_api(monkeypatch: pytest.MonkeyPatch) -> Mock:
    batch = Mock()
    batch.read_namespaced_cron_job.return_value = Mock(
        metadata=client.V1ObjectMeta(name="weather-update", uid="current-uid")
    )
    batch.list_namespaced_job.return_value = client.V1JobList(
        items=[
            client.V1Job(
                metadata=client.V1ObjectMeta(
                    name="self-job",
                    creation_timestamp=datetime(2025, 1, 1, 1, tzinfo=UTC),
                )
            )
        ]
    )
    monkeypatch.setenv("POD_NAMESPACE", "weather")
    monkeypatch.setattr(
        "reformatters.common.kubernetes.config.load_incluster_config", Mock()
    )
    monkeypatch.setattr(
        "reformatters.common.kubernetes.client.BatchV1Api", lambda: batch
    )
    return batch


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize(
    "status",
    [
        pytest.param(None, id="no-status"),
        pytest.param(client.V1JobStatus(), id="pending"),
        pytest.param(client.V1JobStatus(active=1), id="active"),
        pytest.param(client.V1JobStatus(active=0, failed=3), id="retrying"),
    ],
)
def test_has_running_cronjob_job_without_name_detects_any_unfinished_target(
    running_cronjob_api: Mock,
    monkeypatch: pytest.MonkeyPatch,
    owned: bool,
    status: client.V1JobStatus | None,
) -> None:
    labels = {
        "dynamical.org/cronjob-name": "weather-update",
        "dynamical.org/cronjob-uid": "current-uid",
    }
    owners = [
        client.V1OwnerReference(
            api_version="batch/v1",
            kind="CronJob",
            name="weather-update",
            uid="current-uid",
        )
    ]
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name="self-job",
            labels=None if owned else labels,
            owner_references=owners if owned else None,
        ),
        status=status,
    )
    running_cronjob_api.list_namespaced_job.return_value = client.V1JobList(items=[job])
    retry_name = Mock(
        side_effect=AssertionError("Unnamed guard must not check retry parents")
    )
    monkeypatch.setattr("reformatters.common.kubernetes.retry_job_name", retry_name)

    assert has_running_cronjob_job("weather-update")
    retry_name.assert_not_called()
    running_cronjob_api.read_namespaced_cron_job.assert_called_once_with(
        "weather-update", "weather", _request_timeout=10
    )
    running_cronjob_api.list_namespaced_job.assert_called_once_with(
        "weather", _request_timeout=10
    )


@pytest.mark.parametrize("condition_type", ["Complete", "Failed"])
def test_has_running_cronjob_job_without_name_ignores_terminal_targets(
    running_cronjob_api: Mock,
    condition_type: str,
) -> None:
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name="target-job",
            labels={
                "dynamical.org/cronjob-name": "weather-update",
                "dynamical.org/cronjob-uid": "current-uid",
            },
        ),
        status=client.V1JobStatus(
            active=1,
            conditions=[client.V1JobCondition(type=condition_type, status="True")],
        ),
    )
    running_cronjob_api.list_namespaced_job.return_value = client.V1JobList(items=[job])
    assert not has_running_cronjob_job("weather-update")


@pytest.mark.parametrize("owned", [False, True])
def test_has_running_cronjob_job_without_name_excludes_foreign_uid(
    running_cronjob_api: Mock,
    owned: bool,
) -> None:
    labels = {
        "dynamical.org/cronjob-name": "weather-update",
        "dynamical.org/cronjob-uid": "foreign-uid",
    }
    owners = [
        client.V1OwnerReference(
            api_version="batch/v1",
            kind="CronJob",
            name="weather-update",
            uid="foreign-uid",
        )
    ]
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name="target-job",
            labels=None if owned else labels,
            owner_references=owners if owned else None,
        ),
        status=client.V1JobStatus(active=1),
    )
    running_cronjob_api.list_namespaced_job.return_value = client.V1JobList(items=[job])
    assert not has_running_cronjob_job("weather-update")


def test_has_running_cronjob_job_without_name_allows_empty_namespace(
    running_cronjob_api: Mock,
) -> None:
    running_cronjob_api.list_namespaced_job.return_value = client.V1JobList(items=[])
    assert not has_running_cronjob_job("weather-update", None)


@pytest.mark.parametrize(
    ("labels", "owner_kind", "owner_name", "owner_uid", "expected"),
    [
        pytest.param(
            None, "CronJob", "weather-update", "current-uid", True, id="owned"
        ),
        pytest.param(
            {
                "dynamical.org/cronjob-name": "weather-update",
                "dynamical.org/cronjob-uid": "current-uid",
            },
            None,
            None,
            None,
            True,
            id="manual",
        ),
        pytest.param(
            {
                "dynamical.org/cronjob-name": "foreign-update",
                "dynamical.org/cronjob-uid": "current-uid",
            },
            None,
            None,
            None,
            False,
            id="foreign-label",
        ),
        pytest.param(
            {
                "dynamical.org/cronjob-name": "weather-update",
                "dynamical.org/cronjob-uid": "stale-uid",
            },
            None,
            None,
            None,
            False,
            id="stale-label",
        ),
        pytest.param(
            {"dynamical.org/cronjob-name": "weather-update"},
            None,
            None,
            None,
            False,
            id="missing-uid-label",
        ),
        pytest.param(
            {"dynamical.org/cronjob-uid": "current-uid"},
            None,
            None,
            None,
            False,
            id="missing-name-label",
        ),
        pytest.param(
            None, "CronJob", "foreign-update", "current-uid", False, id="foreign-owner"
        ),
        pytest.param(
            None, "CronJob", "weather-update", "stale-uid", False, id="stale-owner"
        ),
        pytest.param(
            None, "Job", "weather-update", "current-uid", False, id="foreign-owner-kind"
        ),
        pytest.param(None, None, None, None, False, id="unrelated-name-prefix"),
    ],
)
def test_has_running_cronjob_job_matches_current_provenance(
    running_cronjob_api: Mock,
    monkeypatch: pytest.MonkeyPatch,
    labels: dict[str, str] | None,
    owner_kind: str | None,
    owner_name: str | None,
    owner_uid: str | None,
    expected: bool,
) -> None:
    owners = (
        [
            client.V1OwnerReference(
                api_version="batch/v1", kind=owner_kind, name=owner_name, uid=owner_uid
            )
        ]
        if owner_kind
        else None
    )
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name="weather-update-other",
            labels=labels,
            owner_references=owners,
            creation_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        status=client.V1JobStatus(active=1),
    )
    running_cronjob_api.list_namespaced_job.return_value.items.append(job)
    log = Mock()
    monkeypatch.setattr("reformatters.common.kubernetes.log", log)

    assert has_running_cronjob_job("weather-update", "self-job") is expected
    running_cronjob_api.read_namespaced_cron_job.assert_called_once_with(
        "weather-update", "weather", _request_timeout=10
    )
    running_cronjob_api.list_namespaced_job.assert_called_once_with(
        "weather", _request_timeout=10
    )
    if expected:
        log.info.assert_called_once_with(
            "CronJob weather-update has unfinished Job weather-update-other"
        )
    else:
        log.info.assert_not_called()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(None, True, id="no-status"),
        pytest.param(client.V1JobStatus(), True, id="pending"),
        pytest.param(client.V1JobStatus(active=0), True, id="zero-active"),
        pytest.param(client.V1JobStatus(active=0, failed=3), True, id="retrying"),
        pytest.param(
            client.V1JobStatus(active=0, succeeded=1),
            True,
            id="succeeded-without-terminal-condition",
        ),
        *[
            pytest.param(
                client.V1JobStatus(
                    active=0,
                    conditions=[
                        client.V1JobCondition(
                            type=condition_type, status=condition_status
                        )
                    ],
                ),
                True,
                id=f"{condition_type}-{condition_status}",
            )
            for condition_type, condition_status in (
                ("Complete", "False"),
                ("Failed", "False"),
                ("Complete", "Unknown"),
                ("Failed", "Unknown"),
                ("SuccessCriteriaMet", "True"),
                ("FailureTarget", "True"),
            )
        ],
        *[
            pytest.param(
                client.V1JobStatus(
                    active=1,
                    conditions=[
                        client.V1JobCondition(type=condition_type, status="True")
                    ],
                ),
                False,
                id=f"terminal-{condition_type}",
            )
            for condition_type in ("Complete", "Failed")
        ],
    ],
)
def test_has_running_cronjob_job_uses_terminal_conditions(
    running_cronjob_api: Mock, status: client.V1JobStatus | None, expected: bool
) -> None:
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name="other-job",
            creation_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            labels={
                "dynamical.org/cronjob-name": "weather-update",
                "dynamical.org/cronjob-uid": "current-uid",
            },
        ),
        status=status,
    )
    running_cronjob_api.list_namespaced_job.return_value.items.append(job)
    assert has_running_cronjob_job("weather-update", "self-job") is expected


@pytest.mark.parametrize("manual", [False, True])
def test_has_running_cronjob_job_excludes_self(
    running_cronjob_api: Mock, manual: bool
) -> None:
    labels = (
        {
            "dynamical.org/cronjob-name": "weather-update",
            "dynamical.org/cronjob-uid": "current-uid",
        }
        if manual
        else None
    )
    owners = (
        None
        if manual
        else [
            client.V1OwnerReference(
                api_version="batch/v1",
                kind="CronJob",
                name="weather-update",
                uid="current-uid",
            )
        ]
    )
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name="self-job",
            labels=labels,
            owner_references=owners,
            creation_timestamp=datetime(2025, 1, 1, 1, tzinfo=UTC),
        ),
        status=client.V1JobStatus(active=1),
    )
    running_cronjob_api.list_namespaced_job.return_value = client.V1JobList(items=[job])
    assert not has_running_cronjob_job("weather-update", "self-job")


def test_has_running_cronjob_job_ignores_younger_unfinished_jobs(
    running_cronjob_api: Mock,
) -> None:
    labels = {
        "dynamical.org/cronjob-name": "weather-update",
        "dynamical.org/cronjob-uid": "current-uid",
    }
    jobs = [
        *running_cronjob_api.list_namespaced_job.return_value.items,
        client.V1Job(
            metadata=client.V1ObjectMeta(
                name="finished-job",
                labels=labels,
                creation_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            ),
            status=client.V1JobStatus(
                conditions=[client.V1JobCondition(type="Complete", status="True")]
            ),
        ),
        client.V1Job(metadata=client.V1ObjectMeta(name="unrelated-job")),
        client.V1Job(
            metadata=client.V1ObjectMeta(
                name="newer-job",
                labels=labels,
                creation_timestamp=datetime(2025, 1, 1, 2, tzinfo=UTC),
            )
        ),
    ]
    running_cronjob_api.list_namespaced_job.return_value = client.V1JobList(items=jobs)
    assert not has_running_cronjob_job("weather-update", "self-job")
    running_cronjob_api.list_namespaced_job.assert_called_once_with(
        "weather", _request_timeout=10
    )


def test_has_running_cronjob_job_without_siblings(running_cronjob_api: Mock) -> None:
    assert not has_running_cronjob_job("weather-update", "self-job")


@pytest.mark.parametrize("with_sibling", [False, True])
def test_has_running_cronjob_job_requires_self_in_list(
    running_cronjob_api: Mock, with_sibling: bool
) -> None:
    jobs = (
        [
            client.V1Job(
                metadata=client.V1ObjectMeta(
                    name="older-job",
                    labels={
                        "dynamical.org/cronjob-name": "weather-update",
                        "dynamical.org/cronjob-uid": "current-uid",
                    },
                    creation_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                )
            )
        ]
        if with_sibling
        else []
    )
    running_cronjob_api.list_namespaced_job.return_value = client.V1JobList(items=jobs)
    with pytest.raises(
        AssertionError, match="Job self-job not found in namespace weather"
    ):
        has_running_cronjob_job("weather-update", "self-job")
    running_cronjob_api.list_namespaced_job.assert_called_once_with(
        "weather", _request_timeout=10
    )


@pytest.mark.parametrize("reverse_listing", [False, True])
@pytest.mark.parametrize(
    ("older_name", "younger_name", "delay"),
    [
        pytest.param(
            "older-job", "younger-job", timedelta(seconds=1), id="older-timestamp"
        ),
        pytest.param("a-job", "b-job", timedelta(0), id="same-timestamp-name-tie"),
        pytest.param(
            "z-job", "a-job", timedelta(seconds=1), id="timestamp-before-name"
        ),
    ],
)
def test_has_running_cronjob_job_concurrent_pair_only_younger_defers(
    running_cronjob_api: Mock,
    reverse_listing: bool,
    older_name: str,
    younger_name: str,
    delay: timedelta,
) -> None:
    labels = {
        "dynamical.org/cronjob-name": "weather-update",
        "dynamical.org/cronjob-uid": "current-uid",
    }
    created = datetime(2025, 1, 1, tzinfo=UTC)
    jobs = [
        client.V1Job(
            metadata=client.V1ObjectMeta(
                name=older_name, labels=labels, creation_timestamp=created
            )
        ),
        client.V1Job(
            metadata=client.V1ObjectMeta(
                name=younger_name, labels=labels, creation_timestamp=created + delay
            )
        ),
    ]
    running_cronjob_api.list_namespaced_job.return_value = client.V1JobList(
        items=list(reversed(jobs)) if reverse_listing else jobs
    )
    assert not has_running_cronjob_job("weather-update", older_name)
    running_cronjob_api.list_namespaced_job.assert_called_once_with(
        "weather", _request_timeout=10
    )
    running_cronjob_api.reset_mock()
    assert has_running_cronjob_job("weather-update", younger_name)
    running_cronjob_api.list_namespaced_job.assert_called_once_with(
        "weather", _request_timeout=10
    )


@pytest.mark.parametrize("unrelated_older", [False, True])
@pytest.mark.parametrize(
    "parent_name",
    [
        pytest.param("weather-update", id="initial-parent"),
        pytest.param("weather-update-r1", id="retry-parent"),
        pytest.param("a" * 62 + "b", id="long-digested-parent"),
    ],
)
def test_has_running_cronjob_job_ignores_active_retry_parent_only(
    running_cronjob_api: Mock,
    monkeypatch: pytest.MonkeyPatch,
    parent_name: str,
    unrelated_older: bool,
) -> None:
    _, own_name = retry_job_name(parent_name)
    if len(parent_name) > 52:
        assert len(own_name) == 52
        assert own_name != parent_name + "-r1"
    own_job = running_cronjob_api.list_namespaced_job.return_value.items[0]
    own_job.metadata.name = own_name
    labels = {
        "dynamical.org/cronjob-name": "weather-update",
        "dynamical.org/cronjob-uid": "current-uid",
    }
    parent = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=parent_name,
            labels=labels,
            creation_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        status=client.V1JobStatus(active=1),
    )
    jobs = [parent, own_job]
    if unrelated_older:
        jobs.append(
            client.V1Job(
                metadata=client.V1ObjectMeta(
                    name="unrelated-older-job",
                    labels=labels,
                    creation_timestamp=datetime(2024, 12, 31, tzinfo=UTC),
                ),
                status=client.V1JobStatus(active=1),
            )
        )
    running_cronjob_api.list_namespaced_job.return_value = client.V1JobList(items=jobs)
    log = Mock()
    monkeypatch.setattr("reformatters.common.kubernetes.log", log)

    assert has_running_cronjob_job("weather-update", own_name) is unrelated_older
    running_cronjob_api.read_namespaced_cron_job.assert_called_once_with(
        "weather-update", "weather", _request_timeout=10
    )
    running_cronjob_api.list_namespaced_job.assert_called_once_with(
        "weather", _request_timeout=10
    )
    if unrelated_older:
        log.info.assert_called_once_with(
            "CronJob weather-update has unfinished Job unrelated-older-job"
        )
    else:
        log.info.assert_not_called()


def test_has_running_cronjob_job_requires_current_uid(
    running_cronjob_api: Mock,
) -> None:
    running_cronjob_api.read_namespaced_cron_job.return_value.metadata.uid = None
    with pytest.raises(AssertionError, match="CronJob has no UID"):
        has_running_cronjob_job("weather-update", "self-job")
    running_cronjob_api.list_namespaced_job.assert_not_called()


@pytest.mark.parametrize(
    "operation", ["read_namespaced_cron_job", "list_namespaced_job"]
)
def test_has_running_cronjob_job_propagates_api_errors(
    running_cronjob_api: Mock, operation: str
) -> None:
    request = getattr(running_cronjob_api, operation)
    request.side_effect = ApiException(status=403)
    with pytest.raises(ApiException) as error:
        has_running_cronjob_job("weather-update", "self-job")
    assert error.value.status == 403
    assert request.call_count == 1


def _deployed_cronjob(*, suspend: bool = False) -> Mock:
    cronjob = Mock()
    cronjob.metadata.uid = "12345678-1234-1234-1234-123456789abc"
    cronjob.spec.suspend = suspend
    cronjob.spec.schedule = "0 * * * *"
    cronjob.spec.job_template.metadata = client.V1ObjectMeta(
        labels={"template-label": "value"}, annotations={"template-note": "value"}
    )
    cronjob.spec.job_template.spec = client.V1JobSpec(
        template=client.V1PodTemplateSpec(
            spec=client.V1PodSpec(
                containers=[client.V1Container(name="worker")],
                active_deadline_seconds=600,
            )
        )
    )
    return cronjob


def test_create_job_clones_live_template_without_owner_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cronjob = _deployed_cronjob()
    batch = Mock()
    batch.read_namespaced_cron_job.return_value = cronjob
    load_config = Mock()
    monkeypatch.setenv("POD_NAMESPACE", "weather")
    monkeypatch.setattr(
        "reformatters.common.kubernetes.config.load_incluster_config", load_config
    )
    monkeypatch.setattr(
        "reformatters.common.kubernetes.client.BatchV1Api", lambda: batch
    )

    assert create_job_from_cronjob("weather-update", "weather-retry-r1")
    load_config.assert_called_once_with()
    batch.read_namespaced_cron_job.assert_called_once_with(
        "weather-update", "weather", _request_timeout=10
    )
    batch.create_namespaced_job.assert_called_once()
    namespace, job = batch.create_namespaced_job.call_args.args
    assert batch.create_namespaced_job.call_args.kwargs == {"_request_timeout": 10}
    assert namespace == "weather"
    assert job.metadata.name == "weather-retry-r1"
    assert job.metadata.owner_references is None
    assert job.metadata.labels == {
        "template-label": "value",
        "dynamical.org/cronjob-name": "weather-update",
        "dynamical.org/cronjob-uid": cronjob.metadata.uid,
    }
    assert job.metadata.annotations == {
        "template-note": "value",
        "cronjob.kubernetes.io/instantiate": "manual",
    }
    assert job.spec == cronjob.spec.job_template.spec
    assert job.spec is not cronjob.spec.job_template.spec


@pytest.mark.parametrize(
    ("minutes_until_fire", "created"), [(11, True), (9, False), (10, False)]
)
def test_create_job_skips_only_within_live_deadline(
    monkeypatch: pytest.MonkeyPatch, minutes_until_fire: int, created: bool
) -> None:
    cronjob = _deployed_cronjob()
    batch = Mock()
    batch.read_namespaced_cron_job.return_value = cronjob
    monkeypatch.setenv("POD_NAMESPACE", "weather")
    monkeypatch.setattr(
        "reformatters.common.kubernetes.config.load_incluster_config", Mock()
    )
    monkeypatch.setattr(
        "reformatters.common.kubernetes.client.BatchV1Api", lambda: batch
    )
    monkeypatch.setattr(
        "reformatters.common.kubernetes.datetime",
        Mock(
            now=lambda tz: (
                datetime(2026, 8, 2, 13, 0, tzinfo=UTC)
                - timedelta(minutes=minutes_until_fire)
            )
        ),
    )

    assert (
        create_job_from_cronjob(
            "weather-update", "weather-retry-r1", skip_if_next_run_within_deadline=True
        )
        is created
    )
    assert batch.create_namespaced_job.called is created
    batch.read_namespaced_cron_job.assert_called_once_with(
        "weather-update", "weather", _request_timeout=10
    )
    if created:
        assert batch.create_namespaced_job.call_args.kwargs == {"_request_timeout": 10}


def test_create_job_respects_suspend(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = Mock()
    batch.read_namespaced_cron_job.return_value = _deployed_cronjob(suspend=True)
    monkeypatch.setenv("POD_NAMESPACE", "weather")
    monkeypatch.setattr(
        "reformatters.common.kubernetes.config.load_incluster_config", Mock()
    )
    monkeypatch.setattr(
        "reformatters.common.kubernetes.client.BatchV1Api", lambda: batch
    )

    assert not create_job_from_cronjob("weather-update", "weather-retry-r1")
    batch.read_namespaced_cron_job.assert_called_once_with(
        "weather-update", "weather", _request_timeout=10
    )
    batch.create_namespaced_job.assert_not_called()


@pytest.mark.parametrize("job_name", ["A-job", "job_name", "-job", "a" * 64])
def test_create_job_rejects_invalid_name(job_name: str) -> None:
    with pytest.raises(AssertionError, match="Invalid Kubernetes Job name"):
        create_job_from_cronjob("weather-update", job_name)


@pytest.mark.parametrize("matching", [True, False])
def test_create_job_409_requires_matching_identity(
    monkeypatch: pytest.MonkeyPatch, matching: bool
) -> None:
    cronjob = _deployed_cronjob()
    batch = Mock()
    batch.read_namespaced_cron_job.return_value = cronjob
    batch.create_namespaced_job.side_effect = ApiException(status=409)
    batch.read_namespaced_job.return_value = client.V1Job(
        metadata=client.V1ObjectMeta(
            name="weather-retry-r1",
            labels={
                "dynamical.org/cronjob-name": "weather-update",
                "dynamical.org/cronjob-uid": (
                    cronjob.metadata.uid if matching else "different-uid"
                ),
            },
        )
    )
    monkeypatch.setenv("POD_NAMESPACE", "weather")
    monkeypatch.setattr(
        "reformatters.common.kubernetes.config.load_incluster_config", Mock()
    )
    monkeypatch.setattr(
        "reformatters.common.kubernetes.client.BatchV1Api", lambda: batch
    )

    if matching:
        assert create_job_from_cronjob("weather-update", "weather-retry-r1")
    else:
        with pytest.raises(ApiException) as error:
            create_job_from_cronjob("weather-update", "weather-retry-r1")
        assert error.value.status == 409
    batch.create_namespaced_job.assert_called_once()
    assert batch.create_namespaced_job.call_args.kwargs == {"_request_timeout": 10}
    batch.read_namespaced_job.assert_called_once_with(
        "weather-retry-r1", "weather", _request_timeout=10
    )


def test_create_job_propagates_non_conflict_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Mock()
    batch.read_namespaced_cron_job.return_value = _deployed_cronjob()
    batch.create_namespaced_job.side_effect = ApiException(status=403)
    monkeypatch.setenv("POD_NAMESPACE", "weather")
    monkeypatch.setattr(
        "reformatters.common.kubernetes.config.load_incluster_config", Mock()
    )
    monkeypatch.setattr(
        "reformatters.common.kubernetes.client.BatchV1Api", lambda: batch
    )

    with pytest.raises(ApiException) as error:
        create_job_from_cronjob("weather-update", "weather-retry-r1")
    assert error.value.status == 403
    batch.read_namespaced_job.assert_not_called()


def test_retry_job_name_is_anchored_and_length_safe() -> None:
    assert retry_job_name("forecast-update") == (True, "forecast-update-r1")
    assert retry_job_name("forecast-update-r1") == (False, "forecast-update-r2")
    assert retry_job_name("forecast-update", max_retries=0) == (
        False,
        "forecast-update-r1",
    )
    assert retry_job_name("forecast-r1-update", max_retries=2) == (
        True,
        "forecast-r1-update-r1",
    )
    assert retry_job_name("forecast-update-r1", max_retries=2) == (
        True,
        "forecast-update-r2",
    )
    should_retry, first = retry_job_name("a" * 62 + "b")
    assert should_retry
    should_retry, second = retry_job_name("a" * 62 + "c")
    assert should_retry
    assert len(first) <= 52
    assert len(second) <= 52
    assert len(first.rsplit("-", 2)[1]) == 4
    assert len(f"{first}-2147483646") <= 63
    assert first != second
    assert (True, first) == retry_job_name("a" * 62 + "b")
