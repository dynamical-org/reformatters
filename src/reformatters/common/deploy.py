import json
import subprocess
from collections.abc import Iterable
from typing import Protocol

from reformatters.common import docker, kubernetes
from reformatters.noaa.gefs.forecast_35_day.reformat import (
    operational_kubernetes_resources as noaa_gefs_forecast_35_day_operational_kubernetes_resources,
)


class OperationalKubernetesResources(Protocol):
    def __call__(self, image_tag: str) -> Iterable[kubernetes.Job]: ...


OPERATIONAL_RESOURCE_FNS: tuple[OperationalKubernetesResources] = (
    noaa_gefs_forecast_35_day_operational_kubernetes_resources,
)


def deploy_operational_updates(
    fns: tuple[OperationalKubernetesResources] = OPERATIONAL_RESOURCE_FNS,
    docker_image: str | None = None,
) -> None:
    image_tag = docker_image or docker.build_and_push_image()

    reformat_jobs: list[kubernetes.Job] = []
    for fn in fns:
        reformat_jobs.extend(fn(image_tag))

    k8s_resource_list = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            reformat_job.as_kubernetes_object() for reformat_job in reformat_jobs
        ],
    }

    subprocess.run(  # noqa: S603
        ["/usr/bin/kubectl", "apply", "-f", "-"],
        input=json.dumps(k8s_resource_list),
        text=True,
        check=True,
    )
