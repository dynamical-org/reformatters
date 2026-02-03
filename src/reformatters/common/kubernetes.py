import base64
import json
import os
import random
import string
from collections.abc import Sequence
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any

import pydantic
from kubernetes import client, config

from reformatters.common.config import Config

_SECRET_MOUNT_PATH = "/secrets"  # noqa: S105
_SECRET_CONTENTS_KEY = "contents"  # noqa: S105


class Job(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    command: Annotated[Sequence[str], pydantic.Field(min_length=1)]
    image: Annotated[str, pydantic.Field(min_length=1)]
    dataset_id: Annotated[str, pydantic.Field(min_length=1)]

    cpu: Annotated[str, pydantic.Field(min_length=1)]
    memory: Annotated[str, pydantic.Field(min_length=1)]
    shared_memory: Annotated[str | None, pydantic.Field(min_length=1)] = None
    ephemeral_storage: Annotated[str, pydantic.Field(min_length=1)] = "10G"

    workers_total: Annotated[int, pydantic.Field(ge=1)]
    parallelism: Annotated[int, pydantic.Field(ge=1)]

    pod_active_deadline: timedelta = timedelta(hours=6)
    ttl: timedelta = timedelta(days=1)

    secret_names: Sequence[str] = pydantic.Field(default_factory=list)

    @cached_property
    def job_name(self) -> str:
        # Job names should be a valid DNS name, 63 characters or less
        name = f"{self.dataset_id[:21]}-{'-'.join(self.command)}"
        name = name.lower().replace("_", "-").replace(":", "-")
        # we add 5 random, then pods within the job add 9. 49+5+9=63
        name = name[:49].rstrip("-")
        random_chars = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=4)  # noqa: S311
        )
        return f"{name}-{random_chars}"

    def as_kubernetes_object(self) -> dict[str, Any]:
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": self.job_name},
            "spec": {
                "backoffLimitPerIndex": 5,
                "completionMode": "Indexed",
                "completions": self.workers_total,
                # A low numbers of workers max == total, then max = total // 8, finally max = 100
                "maxFailedIndexes": min(
                    100, max(min(5, self.workers_total), self.workers_total // 8)
                ),
                "parallelism": self.parallelism,
                "podFailurePolicy": {
                    "rules": [
                        {
                            "action": "Ignore",
                            "onPodConditions": [
                                {"type": "DisruptionTarget", "status": "True"}
                            ],
                        },
                        {
                            "action": "FailJob",
                            "onPodConditions": [
                                {"type": "ConfigIssue", "status": "True"}
                            ],
                        },
                    ]
                },
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "command": [
                                    "python",
                                    "src/reformatters/__main__.py",
                                    f"{self.dataset_id}",
                                    *self.command,
                                ],
                                "env": [
                                    {"name": "DYNAMICAL_ENV", "value": "prod"},
                                    {
                                        "name": "DYNAMICAL_SENTRY_DSN",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "key": "DYNAMICAL_SENTRY_DSN",
                                                "name": "sentry",
                                            }
                                        },
                                    },
                                    {
                                        "name": "JOB_NAME",
                                        "valueFrom": {
                                            "fieldRef": {
                                                "fieldPath": "metadata.labels['job-name']"
                                            }
                                        },
                                    },
                                    {
                                        "name": "POD_NAME",
                                        "valueFrom": {
                                            "fieldRef": {"fieldPath": "metadata.name"}
                                        },
                                    },
                                    {
                                        "name": "WORKER_INDEX",
                                        "valueFrom": {
                                            "fieldRef": {
                                                "fieldPath": "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                                            }
                                        },
                                    },
                                    {
                                        "name": "WORKERS_TOTAL",
                                        "value": f"{self.workers_total}",
                                    },
                                ],
                                "image": f"{self.image}",
                                "name": "worker",
                                "resources": {
                                    "requests": {
                                        "cpu": f"{self.cpu}",
                                        "memory": f"{self.memory}",
                                    }
                                },
                                "volumeMounts": [
                                    {"mountPath": "/app/data", "name": "ephemeral-vol"},
                                    *(
                                        [
                                            {
                                                "mountPath": "/dev/shm",  # noqa: S108 yes we're using a known, shared path
                                                "name": "shared-memory-dir",
                                            }
                                        ]
                                        if self.shared_memory is not None
                                        else []
                                    ),
                                    *[
                                        {
                                            "name": secret_name,
                                            "mountPath": f"/secrets/{secret_name}.json",
                                            "subPath": _SECRET_CONTENTS_KEY,
                                            "readOnly": True,
                                        }
                                        for secret_name in self.secret_names
                                    ],
                                ],
                            }
                        ],
                        "nodeSelector": {
                            "eks.amazonaws.com/compute-type": "auto",
                            "karpenter.sh/capacity-type": "spot",
                        },
                        "restartPolicy": "Never",
                        "securityContext": {
                            "fsGroup": 999,  # this is the `app` group our app runs under
                        },
                        "terminationGracePeriodSeconds": 5,
                        "activeDeadlineSeconds": int(
                            self.pod_active_deadline.total_seconds()
                        ),
                        "volumes": [
                            {
                                "ephemeral": {
                                    "volumeClaimTemplate": {
                                        "metadata": {"labels": {"type": "ephemeral"}},
                                        "spec": {
                                            "accessModes": ["ReadWriteOnce"],
                                            "resources": {
                                                "requests": {
                                                    "storage": f"{self.ephemeral_storage}"
                                                }
                                            },
                                        },
                                    }
                                },
                                "name": "ephemeral-vol",
                            },
                            *(
                                [
                                    {
                                        "name": "shared-memory-dir",
                                        "emptyDir": {
                                            "medium": "Memory",
                                            "sizeLimit": self.shared_memory,
                                        },
                                    }
                                ]
                                if self.shared_memory is not None
                                else []
                            ),
                            *[
                                {
                                    "name": secret_name,
                                    "secret": {"secretName": secret_name},
                                }
                                for secret_name in self.secret_names
                            ],
                        ],
                    }
                },
                "ttlSecondsAfterFinished": int(self.ttl.total_seconds()),
            },
        }


class CronJob(Job):
    name: Annotated[str, pydantic.Field(min_length=1)]
    schedule: Annotated[str, pydantic.Field(min_length=1)]
    ttl: timedelta = timedelta(hours=12)
    suspend: bool = False

    def as_kubernetes_object(self) -> dict[str, Any]:
        job_spec = super().as_kubernetes_object()["spec"]
        job_spec["template"]["spec"]["containers"][0]["env"].append(
            {
                "name": "CRON_JOB_NAME",
                "value": self.name,
            }
        )
        return {
            "apiVersion": "batch/v1",
            "kind": "CronJob",
            "metadata": {"name": self.name},
            "spec": {
                "schedule": self.schedule,
                "suspend": self.suspend,
                "concurrencyPolicy": "Replace",
                "jobTemplate": {"spec": job_spec},
            },
        }


class ReformatCronJob(CronJob):
    command: Sequence[str] = ["update"]
    # Operational updates expect a single worker
    workers_total: int = 1
    parallelism: int = 1


class ValidationCronJob(CronJob):
    command: Sequence[str] = ["validate"]
    workers_total: int = 1
    parallelism: int = 1


def load_secret(secret_name: str) -> dict[str, Any]:
    """
    Load a secret from kubernetes, either from mounted file or directly from kubernetes API.

    Returns empty dict in non-prod environments.
    When env is prod, loads from mounted secret file, or falls back to kubernetes API if running locally.
    """
    if not Config.is_prod:
        return {}

    secret_file = Path(_SECRET_MOUNT_PATH) / f"{secret_name}.json"

    if not secret_file.exists():
        if os.getenv("JOB_NAME") is not None:
            # We're in a cluster, the secret should be mounted at the expected path
            raise FileNotFoundError(
                f"Secret file {secret_file} not found in production job"
            )
        else:
            # Local case, e.g. to support backfill-kubernetes writing the zarr metadata
            return _load_secret_from_kubernetes_api(secret_name)

    with open(secret_file) as f:
        contents = json.load(f)
        assert isinstance(contents, dict)
        return contents


def _load_secret_from_kubernetes_api(
    secret_name: str,
) -> dict[str, Any]:
    """Load secret directly from kubernetes API (for local development)."""
    config.load_kube_config()
    v1 = client.CoreV1Api()
    secret = v1.read_namespaced_secret(secret_name, "default")
    assert isinstance(secret.data, dict)
    contents_json = base64.b64decode(secret.data[_SECRET_CONTENTS_KEY]).decode("utf-8")
    contents = json.loads(contents_json)
    assert isinstance(contents, dict)
    return contents
