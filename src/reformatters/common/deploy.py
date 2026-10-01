import json
import subprocess
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import typer

from reformatters.common import docker, kubernetes, staging
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes_security import (
    trigger_admission_resources,
    trigger_deployment_resources,
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

    reformat_jobs: list[kubernetes.Job] = []
    trigger_templates: dict[str, dict[str, Any]] = {}
    deployed_trigger_templates: list[dict[str, Any]] = []

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

        trigger_templates.update(
            (cj.name, cj.as_kubernetes_object())
            for cj in dataset_cronjobs
            if cj.triggerable
        )
        if dataset_id_filter is not None and resource.dataset_id != dataset_id_filter:
            continue

        if cronjob_transform is not None:
            dataset_cronjobs = [cronjob_transform(cj) for cj in dataset_cronjobs]

        deployed_trigger_templates.extend(
            cj.as_kubernetes_object() for cj in dataset_cronjobs if cj.triggerable
        )

        reformat_jobs.extend(dataset_cronjobs)

    assert len(reformat_jobs) > 0, "No cronjobs to deploy" + (
        f" for dataset_id_filter={dataset_id_filter!r}" if dataset_id_filter else ""
    )

    # An archiver-only deploy still grants access to its already-deployed targets.
    templates = deployed_trigger_templates or list(trigger_templates.values())
    verify_trigger_admission("default", templates, require_params=False)

    workloads, permissions = trigger_deployment_resources(reformat_jobs)
    _apply_resources(workloads)
    verify_trigger_admission("default", templates, require_params=True)
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
    def verify_admission(bundle: Path, namespace: str = "default") -> None:
        """Probe admission using server-side dry runs; creates no resources."""
        items = json.loads(bundle.read_text())["items"]
        names = [
            item["spec"]["paramRef"]["name"]
            for item in items
            if "paramRef" in item["spec"]
        ]
        assert items == trigger_admission_resources(namespace, names), (
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
                capture_output=True,
                text=True,
                check=True,
            )
            assert json.loads(installed.stdout)["spec"] == item["spec"], (
                f"Installed {item['kind']} {item['metadata']['name']} differs from bundle"
            )
        templates = [
            {
                "metadata": {"name": name},
                "spec": {
                    "jobTemplate": {
                        "spec": {
                            "template": {
                                "spec": {
                                    "restartPolicy": "Never",
                                    "containers": [
                                        {
                                            "name": "worker",
                                            "image": "invalid.example/admission-canary:never-run",
                                        }
                                    ],
                                }
                            }
                        }
                    }
                },
            }
            for name in names
        ]
        verify_trigger_admission(namespace, templates, require_params=False)

    @app.command()
    def render_admission_bundle(
        namespace: str = "default",
        staging_target: list[str] | None = None,
    ) -> None:
        """Render cluster-admin admission resources without contacting Kubernetes."""
        targets = set(staging_target or [])
        for resource in [*datasets, *archivers]:
            try:
                cronjobs = resource.operational_kubernetes_resources("unused")
            except NotImplementedError:
                continue
            targets.update(cj.name for cj in cronjobs if cj.triggerable)
        typer.echo(
            json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "List",
                    "items": trigger_admission_resources(namespace, sorted(targets)),
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
