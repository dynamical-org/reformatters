import json
import subprocess
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import typer
import uvicorn

from reformatters.common import docker, kubernetes, staging
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.logging import get_logger
from reformatters.common.webhook import deploy as webhook_deploy
from reformatters.common.webhook.receiver import create_app
from reformatters.common.webhook.subscription import register_subscription

log = get_logger(__name__)


def deploy_operational_resources(
    datasets: Iterable[DynamicalDataset[Any, Any]],
    docker_image: str | None = None,
    dataset_id_filter: str | None = None,
    cronjob_transform: Callable[[kubernetes.CronJob], kubernetes.CronJob] | None = None,
    include_webhook_receiver: bool = False,
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

    items: list[dict[str, Any]] = [
        reformat_job.as_kubernetes_object() for reformat_job in reformat_jobs
    ]
    if include_webhook_receiver:
        items.extend(webhook_deploy.webhook_receiver_resources(image_tag))

    k8s_resource_list = {
        "apiVersion": "v1",
        "kind": "List",
        "items": items,
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


def register_commands(
    app: typer.Typer, datasets: Sequence[DynamicalDataset[Any, Any]]
) -> None:
    @app.command()
    def deploy(
        docker_image: str | None = None,
    ) -> None:
        deploy_operational_resources(
            datasets, docker_image, include_webhook_receiver=True
        )

    @app.command()
    def serve_webhooks(
        host: str = "0.0.0.0",  # noqa: S104  in-cluster service behind an Ingress
        port: int = 8080,
    ) -> None:
        """Run the wxopticon webhook receiver (see docs/webhooks.md)."""
        uvicorn.run(create_app(list(datasets)), host=host, port=port)

    @app.command()
    def register_webhook_subscription(
        webhook_url: str,
        subscription_id: str | None = None,
    ) -> None:
        """Register (or --subscription-id to update) the wxopticon subscription that
        delivers source-arrival webhooks to the receiver. Needs WXOPTICON_ADMIN_TOKEN."""
        result = register_subscription(
            list(datasets), webhook_url, subscription_id=subscription_id
        )
        log.info(f"Subscription id: {result.get('id')}")
        secret = result.get("secret")
        if secret:
            log.info(
                "Store this secret (shown once) in the 'wxopticon-webhook' k8s secret "
                "under key WXOPTICON_WEBHOOK_SECRET:"
            )
            typer.echo(secret)

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
