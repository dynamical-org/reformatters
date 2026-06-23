import json
import os
import subprocess
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import typer

from reformatters.common import betterstack, docker, kubernetes, staging
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.logging import get_logger

log = get_logger(__name__)


def deploy_operational_resources(
    datasets: Iterable[DynamicalDataset[Any, Any]],
    docker_image: str | None = None,
    dataset_id_filter: str | None = None,
    cronjob_transform: Callable[[kubernetes.CronJob], kubernetes.CronJob] | None = None,
) -> None:
    image_tag = docker_image or docker.build_and_push_image()

    reformat_jobs: list[kubernetes.Job] = []

    for dataset in datasets:
        if dataset_id_filter is not None and dataset.dataset_id != dataset_id_filter:
            continue

        try:
            dataset_cronjobs = list(dataset.operational_kubernetes_resources(image_tag))
        except NotImplementedError:
            log.info(
                f"Skipping deploy for {dataset.__class__.__name__}, "
                "`operational_kubernetes_resources` not implemented."
            )
            continue

        if cronjob_transform is not None:
            dataset_cronjobs = [cronjob_transform(cj) for cj in dataset_cronjobs]

        reformat_jobs.extend(dataset_cronjobs)

    assert len(reformat_jobs) > 0, "No cronjobs to deploy" + (
        f" for dataset_id_filter={dataset_id_filter!r}" if dataset_id_filter else ""
    )

    _provision_heartbeats(
        cj for cj in reformat_jobs if isinstance(cj, kubernetes.CronJob)
    )

    k8s_resource_list = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            reformat_job.as_kubernetes_object() for reformat_job in reformat_jobs
        ],
    }

    subprocess.run(
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=json.dumps(k8s_resource_list),
        text=True,
        check=True,
    )

    log.info(
        f"Deployed {[item['metadata']['name'] for item in k8s_resource_list['items']]}"  # type: ignore[index]
    )


def _provision_heartbeats(cron_jobs: Iterable[kubernetes.CronJob]) -> None:
    if os.getenv("DYNAMICAL_BETTERSTACK_UPTIME_API_TOKEN") is None:
        log.warning(
            "DYNAMICAL_BETTERSTACK_UPTIME_API_TOKEN unset; skipping Better Stack "
            "heartbeat provisioning. Cron monitoring will not be updated."
        )
        return
    url_map = betterstack.reconcile_heartbeats(cron_jobs)
    betterstack.write_heartbeat_secret(url_map)


def register_commands(
    app: typer.Typer, datasets: Sequence[DynamicalDataset[Any, Any]]
) -> None:
    @app.command()
    def deploy(
        docker_image: str | None = None,
    ) -> None:
        deploy_operational_resources(datasets, docker_image)

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
