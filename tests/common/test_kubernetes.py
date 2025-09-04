from datetime import timedelta
from typing import Any

import pytest

from reformatters.common.kubernetes import Job


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
            ]
        },
        "template": {
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
                        "envFrom": [],
                        "image": "weather-app:v1.0",
                        "name": "worker",
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
                                "subPath": "storage_options.json",
                                "readOnly": True,
                            },
                            {
                                "name": "db-creds",
                                "mountPath": "/secrets/db-creds.json",
                                "subPath": "storage_options.json",
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
                },
                "terminationGracePeriodSeconds": 5,
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
            }
        },
        "ttlSecondsAfterFinished": 86400,  # default 24 hours
    }

    assert k8s_obj["spec"] == expected_spec


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
    "workers_total,expected_max_failed_indexes",
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
