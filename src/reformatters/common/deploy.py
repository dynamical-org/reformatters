import json
import subprocess
from collections.abc import Iterable
from typing import Any

from reformatters.common import docker, kubernetes
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.logging import get_logger

log = get_logger(__name__)


def deploy_operational_updates(
    datasets: Iterable[DynamicalDataset[Any, Any]],
    docker_image: str | None = None,
) -> None:
    image_tag = docker_image or docker.build_and_push_image()

    reformat_jobs: list[kubernetes.Job] = []

    for dataset in datasets:
        try:
            reformat_jobs.extend(dataset.operational_kubernetes_resources(image_tag))
        except NotImplementedError:
            log.info(
                f"Skipping deploy for {dataset.__class__.__name__}, "
                "`operational_kubernetes_resources` not implemented."
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
