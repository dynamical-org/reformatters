from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from kubernetes import client, config

from reformatters.common.kubernetes import CronJob, Job
from reformatters.common.operational import OperationalResources


@dataclass(frozen=True)
class Cluster:
    name: str
    resources: Sequence[OperationalResources]
    capacity_type: Literal["spot", "on-demand"]
    suspended: bool = False
    github_environment: str = "prod"
    enabled: bool = True

    def configure_job[J: Job](self, job: J) -> J:
        configured = job.model_copy(
            update={"capacity_type": self.capacity_type, "cluster": self.name}
        )
        if isinstance(configured, CronJob):
            configured.suspend = configured.suspend or self.suspended
        return configured


def clusters() -> tuple[Cluster, ...]:
    from reformatters.deployments import aws_us_east_1, aws_us_west_2  # noqa: PLC0415

    return (
        Cluster(
            "aws-us-west-2",
            [
                *aws_us_west_2.CLUSTER_DYNAMICAL_DATASETS,
                *aws_us_west_2.CLUSTER_OPERATIONAL_ARCHIVERS,
            ],
            "spot",
            github_environment="prod",
        ),
        Cluster(
            "aws-us-east-1",
            [
                *aws_us_east_1.CLUSTER_DYNAMICAL_DATASETS,
                *aws_us_east_1.CLUSTER_OPERATIONAL_ARCHIVERS,
            ],
            "on-demand",
            suspended=True,
            github_environment="prod",
            enabled=True,
        ),
    )


def get_cluster(name: str) -> Cluster:
    matches = [cluster for cluster in clusters() if cluster.name == name]
    assert len(matches) == 1, f"Unknown or duplicate cluster: {name}"
    return matches[0]


def cluster_for_resource(dataset_id: str) -> Cluster:
    matches = [
        cluster
        for cluster in clusters()
        for resource in cluster.resources
        if resource.dataset_id == dataset_id
    ]
    assert len(matches) == 1, f"Expected one cluster owner for {dataset_id}"
    return matches[0]


def cluster_for_cronjob(name: str) -> Cluster:
    matches: list[Cluster] = []
    for cluster in clusters():
        for resource in cluster.resources:
            try:
                jobs = resource.operational_kubernetes_resources("placeholder")
            except NotImplementedError:
                continue
            matches.extend(cluster for job in jobs if job.name == name)
    assert len(matches) == 1, f"Expected one cluster owner for CronJob {name}"
    return matches[0]


def verify_cluster(cluster: Cluster) -> None:
    config.load_kube_config()
    identity = client.CoreV1Api().read_namespaced_config_map(
        "reformatters-cluster-identity", "default"
    )
    assert identity.data, "Cluster identity ConfigMap has no data"
    assert identity.data.get("cluster") == cluster.name, (
        f"Kubernetes context does not identify as {cluster.name}"
    )
