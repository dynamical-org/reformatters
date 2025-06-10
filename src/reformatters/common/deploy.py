import json
import subprocess
from collections.abc import Iterable
from typing import Any, Protocol

from reformatters.common import docker, kubernetes
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.logging import get_logger
from reformatters.noaa.gefs.analysis.reformat import (
    operational_kubernetes_resources as noaa_gefs_analysis_operational_kubernetes_resources,
)
from reformatters.noaa.gefs.forecast_35_day.reformat import (
    operational_kubernetes_resources as noaa_gefs_forecast_35_day_operational_kubernetes_resources,
)

log = get_logger(__name__)


class OperationalKubernetesResources(Protocol):
    def __call__(self, image_tag: str) -> Iterable[kubernetes.Job]: ...


# For datasets which have not yet been implemented as a DynamicalDataset sublcass
LEGACY_OPERATIONAL_RESOURCE_FNS: tuple[OperationalKubernetesResources, ...] = (
    noaa_gefs_forecast_35_day_operational_kubernetes_resources,
    noaa_gefs_analysis_operational_kubernetes_resources,
)


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

    for fn in LEGACY_OPERATIONAL_RESOURCE_FNS:
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

    log.info(
        f"Deployed {[item['metadata']['name'] for item in k8s_resource_list['items']]}"  # type: ignore[index]
    )
