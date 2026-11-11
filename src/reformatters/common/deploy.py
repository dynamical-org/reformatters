import json
import subprocess
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from reformatters.common import docker, kubernetes, staging
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes_security import (
    create_trigger_bindings,
    trigger_admission_resources,
    trigger_deployment_resources,
    trigger_role_targets,
    verify_trigger_admission,
)
from reformatters.common.logging import get_logger
from reformatters.common.operational import OperationalResources
from reformatters.deployments import targets

log = get_logger(__name__)


def deploy_operational_resources(
    resources: Iterable[OperationalResources],
    docker_image: str | None = None,
    dataset_id_filter: str | None = None,
    cronjob_transform: Callable[[kubernetes.CronJob], kubernetes.CronJob] | None = None,
    cluster: targets.Cluster | None = None,
) -> None:
    resources = list(resources)
    if dataset_id_filter is not None:
        resources = [r for r in resources if r.dataset_id == dataset_id_filter]
        assert resources, f"Unknown dataset_id_filter={dataset_id_filter!r}"
    if not resources:
        return
    image_tag = docker_image or docker.build_and_push_image()

    reformat_jobs: list[kubernetes.CronJob] = []

    for resource in resources:
        try:
            dataset_cronjobs = list(
                resource.operational_kubernetes_resources(image_tag)
            )
        except NotImplementedError:
            log.info(
                f"Skipping deploy for {resource.__class__.__name__}, "
                "`operational_kubernetes_resources` not implemented."
            )
            continue

        if cronjob_transform is not None:
            dataset_cronjobs = [cronjob_transform(cj) for cj in dataset_cronjobs]
        if cluster is not None:
            dataset_cronjobs = [cluster.configure_job(cj) for cj in dataset_cronjobs]

        reformat_jobs.extend(dataset_cronjobs)

    assert len(reformat_jobs) > 0, "No cronjobs to deploy" + (
        f" for dataset_id_filter={dataset_id_filter!r}" if dataset_id_filter else ""
    )

    authorized_targets = trigger_role_targets("default")
    workloads, _ = trigger_deployment_resources(reformat_jobs)
    _apply_resources(workloads)
    names = sorted(authorized_targets | {job.name for job in reformat_jobs})
    create_trigger_bindings("default", names)
    _, permissions = trigger_deployment_resources([], names)
    _apply_resources(permissions)

    log.info(
        "Deployed %s", [item["metadata"]["name"] for item in workloads + permissions]
    )


def _apply_resources(resources: list[dict[str, Any]]) -> None:
    subprocess.run(
        ["/usr/bin/kubectl", "apply", "--namespace", "default", "-f", "-"],
        input=json.dumps({"apiVersion": "v1", "kind": "List", "items": resources}),
        text=True,
        check=True,
    )


def register_commands(
    app: typer.Typer,
    datasets: Sequence[DynamicalDataset[Any, Any]],
) -> None:
    @app.command()
    def resolve_cluster(
        dataset_id: str | None = None, cronjob_name: str | None = None
    ) -> None:
        assert (dataset_id is None) != (cronjob_name is None), (
            "Specify exactly one of --dataset-id and --cronjob-name"
        )
        cluster = (
            targets.cluster_for_resource(dataset_id)
            if dataset_id is not None
            else targets.cluster_for_cronjob(str(cronjob_name))
        )
        typer.echo(cluster.name)

    @app.command()
    def verify_cluster(cluster: str) -> None:
        targets.verify_cluster(targets.get_cluster(cluster))

    @app.command()
    def verify_kubernetes_admission(bundle: Path, namespace: str = "default") -> None:
        """Probe admission using server-side dry runs; creates no resources."""
        items = json.loads(bundle.read_text())["items"]
        assert items == trigger_admission_resources(namespace), (
            "Bundle does not match the complete generated policy"
        )
        for item in items:
            installed = subprocess.run(  # noqa: S603
                [
                    "/usr/bin/kubectl",
                    "get",
                    item["kind"],
                    item["metadata"]["name"],
                    "-o",
                    "json",
                ],
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
            assert json.loads(installed.stdout)["spec"] == item["spec"], (
                f"Installed {item['kind']} {item['metadata']['name']} differs from bundle"
            )
        verify_trigger_admission(namespace, sorted(trigger_role_targets(namespace)))

    @app.command()
    def render_kubernetes_admission_bundle(
        namespace: str = "default",
    ) -> None:
        """Render cluster-admin admission resources without contacting Kubernetes."""
        typer.echo(
            json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "List",
                    "items": trigger_admission_resources(namespace),
                },
                indent=2,
            )
        )

    @app.command()
    def deploy(
        cluster: Annotated[str, typer.Option()],
        docker_image: str | None = None,
        dataset_id: str | None = None,
    ) -> None:
        target = targets.get_cluster(cluster)
        if dataset_id is not None:
            assert targets.cluster_for_resource(dataset_id).name == target.name, (
                f"{dataset_id} is not assigned to {target.name}"
            )
        if not target.resources:
            return
        targets.verify_cluster(target)
        deploy_operational_resources(
            target.resources, docker_image, dataset_id_filter=dataset_id, cluster=target
        )

    @app.command()
    def deploy_staging(
        dataset_id: str,
        version: str,
        docker_image: str,
    ) -> None:
        """Deploy staging cronjobs for a single dataset version."""
        dataset = staging.find_dataset(datasets, dataset_id)
        target = targets.cluster_for_resource(dataset_id)
        targets.verify_cluster(target)
        staging.validate_version_matches_template(dataset, version)
        staging.validate_version_differs_from_main(dataset, version)

        def transform(cronjob: kubernetes.CronJob) -> kubernetes.CronJob:
            return staging.rename_cronjob_for_staging(cronjob, dataset_id, version)

        deploy_operational_resources(
            datasets,
            docker_image=docker_image,
            dataset_id_filter=dataset_id,
            cronjob_transform=transform,
            cluster=target,
        )

    @app.command()
    def cleanup_staging(
        dataset_id: str,
        version: str,
        force: bool = False,
    ) -> None:
        """Clean up staging resources: kubernetes cronjobs and git branch."""
        staging.find_dataset(datasets, dataset_id)  # validate dataset_id
        if not force:
            cronjob_names = staging.staging_cronjob_names(dataset_id, version)
            branch = staging.staging_branch_name(dataset_id, version)
            log.info(
                f"Will delete cronjobs {cronjob_names} and branch {branch}. "
                "Run with --force to execute."
            )
            return
        targets.verify_cluster(targets.cluster_for_resource(dataset_id))
        staging.cleanup_staging_resources(dataset_id, version)
