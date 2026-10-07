import json
import subprocess
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

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

log = get_logger(__name__)


def deploy_operational_resources(
    resources: Iterable[OperationalResources],
    docker_image: str | None = None,
    dataset_id_filter: str | None = None,
    cronjob_transform: Callable[[kubernetes.CronJob], kubernetes.CronJob] | None = None,
) -> None:
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

        if dataset_id_filter is not None and resource.dataset_id != dataset_id_filter:
            continue

        if cronjob_transform is not None:
            dataset_cronjobs = [cronjob_transform(cj) for cj in dataset_cronjobs]

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
    archivers: Sequence[OperationalResources] = (),
) -> None:
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
        verify_trigger_admission(namespace, [])

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
        docker_image: str | None = None,
        dataset_id: str | None = None,
    ) -> None:
        deploy_operational_resources(
            [*datasets, *archivers], docker_image, dataset_id_filter=dataset_id
        )

    @app.command()
    def deploy_staging(
        dataset_id: str,
        version: str,
        docker_image: str,
    ) -> None:
        """Deploy staging cronjobs for a single dataset version."""
        dataset = staging.find_dataset(datasets, dataset_id)
        staging.validate_version_matches_template(dataset, version)
        staging.validate_version_differs_from_main(dataset, version)

        def transform(cronjob: kubernetes.CronJob) -> kubernetes.CronJob:
            return staging.rename_cronjob_for_staging(cronjob, dataset_id, version)

        deploy_operational_resources(
            datasets,
            docker_image=docker_image,
            dataset_id_filter=dataset_id,
            cronjob_transform=transform,
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
        staging.cleanup_staging_resources(dataset_id, version)
