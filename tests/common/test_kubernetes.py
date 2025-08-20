from datetime import timedelta
from typing import Any

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

    # Test job spec structure
    spec: dict[str, Any] = k8s_obj["spec"]
    assert spec["completions"] == 4
    assert spec["parallelism"] == 2
    assert spec["completionMode"] == "Indexed"
    assert spec["backoffLimitPerIndex"] == 5

    # Test pod spec structure
    pod_spec: dict[str, Any] = spec["template"]["spec"]
    container: dict[str, Any] = pod_spec["containers"][0]

    # Test container configuration
    expected_command: list[str] = [
        "python",
        "src/reformatters/__main__.py",
        "weather_data",
        "backfill-kubernetes",
        "2025-01-01T00:00:00",
        "1",
        "1",
    ]
    assert container["command"] == expected_command
    assert container["image"] == "weather-app:v1.0"

    # Test resources
    assert container["resources"]["requests"]["cpu"] == "500m"
    assert container["resources"]["requests"]["memory"] == "1Gi"

    # Test environment variables
    env_vars: dict[str, dict[str, Any]] = {env["name"]: env for env in container["env"]}
    assert env_vars["DYNAMICAL_ENV"]["value"] == "prod"
    assert env_vars["WORKERS_TOTAL"]["value"] == "4"

    # Test secret references
    assert len(container["envFrom"]) == 2
    secret_names: set[str] = {ref["secretRef"]["name"] for ref in container["envFrom"]}
    assert secret_names == {"aws-creds", "db-creds"}

    # Test volume mounts
    volume_mounts: dict[str, Any] = {
        mount["name"]: mount for mount in container["volumeMounts"]
    }
    assert volume_mounts["ephemeral-vol"]["mountPath"] == "/app/data"
    assert volume_mounts["shared-memory-dir"]["mountPath"] == "/dev/shm"  # noqa: S108 yes we're using a known, shared path
    assert volume_mounts["aws-creds"]["mountPath"] == "/secrets/aws-creds.json"
    assert volume_mounts["db-creds"]["mountPath"] == "/secrets/db-creds.json"

    # Test volumes
    volumes: dict[str, dict[str, Any]] = {
        vol["name"]: vol for vol in pod_spec["volumes"]
    }

    # Test ephemeral volume
    ephemeral_vol: dict[str, Any] = volumes["ephemeral-vol"]
    assert "ephemeral" in ephemeral_vol
    assert ephemeral_vol["ephemeral"]["volumeClaimTemplate"]["spec"]["accessModes"] == [
        "ReadWriteOnce"
    ]
    assert (
        ephemeral_vol["ephemeral"]["volumeClaimTemplate"]["spec"]["resources"][
            "requests"
        ]["storage"]
        == "10G"
    )  # default value

    # Test shared memory volume
    shared_mem_vol: dict[str, Any] = volumes["shared-memory-dir"]
    assert shared_mem_vol["emptyDir"]["medium"] == "Memory"
    assert shared_mem_vol["emptyDir"]["sizeLimit"] == "1k"  # default value

    # Test secret volumes
    assert "aws-creds" in volumes
    assert volumes["aws-creds"]["secret"]["secretName"] == "aws-creds"
    assert "db-creds" in volumes
    assert volumes["db-creds"]["secret"]["secretName"] == "db-creds"


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

    # Test shared memory volume
    volumes: dict[str, dict[str, Any]] = {
        vol["name"]: vol for vol in pod_spec["volumes"]
    }
    shared_mem_vol: dict[str, Any] = volumes["shared-memory-dir"]
    assert shared_mem_vol["emptyDir"]["sizeLimit"] == "512Mi"

    # Test ephemeral storage
    ephemeral_vol: dict[str, Any] = volumes["ephemeral-vol"]
    storage_request: str = ephemeral_vol["ephemeral"]["volumeClaimTemplate"]["spec"][
        "resources"
    ]["requests"]["storage"]
    assert storage_request == "50G"
