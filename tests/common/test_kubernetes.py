from datetime import timedelta
from typing import Any

from reformatters.common.kubernetes import Job


def test_as_kubernetes_object_basic_structure() -> None:
    """Test that as_kubernetes_object returns the expected basic structure."""
    job = Job(
        command=["update"],
        image="my-image:latest",
        dataset_id="test_dataset",
        cpu="1000m",
        memory="2Gi",
        workers_total=4,
        parallelism=2,
    )

    k8s_obj: dict[str, Any] = job.as_kubernetes_object()

    # Test top-level structure
    assert k8s_obj["apiVersion"] == "batch/v1"
    assert k8s_obj["kind"] == "Job"

    # Test metadata
    assert "name" in k8s_obj["metadata"]
    assert k8s_obj["metadata"]["name"].startswith("test-dataset-update-")

    # Test spec structure
    spec: dict[str, Any] = k8s_obj["spec"]
    assert spec["completions"] == 4
    assert spec["parallelism"] == 2
    assert spec["completionMode"] == "Indexed"
    assert spec["backoffLimitPerIndex"] == 5


def test_as_kubernetes_object_container_config() -> None:
    """Test that the container configuration is correct."""
    job = Job(
        command=["backfill-kubernetes", "2025-01-01T00:00:00", "1", "1"],
        image="weather-app:v1.0",
        dataset_id="weather_data",
        cpu="500m",
        memory="1Gi",
        workers_total=1,
        parallelism=1,
        secret_names=["aws-creds", "db-creds"],
    )

    k8s_obj: dict[str, Any] = job.as_kubernetes_object()
    container: dict[str, Any] = k8s_obj["spec"]["template"]["spec"]["containers"][0]

    # Test command
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

    # Test image
    assert container["image"] == "weather-app:v1.0"

    # Test resources
    assert container["resources"]["requests"]["cpu"] == "500m"
    assert container["resources"]["requests"]["memory"] == "1Gi"

    # Test environment variables
    env_vars: dict[str, dict[str, Any]] = {env["name"]: env for env in container["env"]}
    assert env_vars["DYNAMICAL_ENV"]["value"] == "prod"
    assert env_vars["WORKERS_TOTAL"]["value"] == "1"

    # Test secret references
    assert len(container["envFrom"]) == 2
    secret_names: set[str] = {ref["secretRef"]["name"] for ref in container["envFrom"]}
    assert secret_names == {"aws-creds", "db-creds"}


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
