from datetime import timedelta
from typing import Any

import pydantic


class ReformatJob(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    command: list[str] = pydantic.Field(min_length=1)
    image: str
    dataset_id: str
    cpu: str
    memory: str
    ephemeral_storage: str
    workers_total: int
    parallelism: int
    ttl: timedelta = timedelta(days=7)

    @property
    def job_name(self) -> str:
        return f"{self.dataset_id}-{'-'.join(self.command)}"

    def as_kubernetes_object(self) -> dict[str, Any]:
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": self.job_name},
            "spec": {
                "backoffLimitPerIndex": 4,
                "completionMode": "Indexed",
                "completions": f"{self.workers_total}",
                "maxFailedIndexes": 3,
                "parallelism": f"{self.parallelism}",
                "podFailurePolicy": {
                    "rules": [
                        {
                            "action": "Ignore",
                            "onPodConditions": [{"type": "DisruptionTarget"}],
                        },
                        {
                            "action": "FailJob",
                            "onPodConditions": [{"type": "ConfigIssue"}],
                        },
                    ]
                },
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "command": [
                                    "python",
                                    "main.py",
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
                                    {
                                        "name": "DYNAMICAL_SOURCE_COOP_AWS_ACCESS_KEY_ID",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "key": "AWS_ACCESS_KEY_ID",
                                                "name": "source-coop-key",
                                            }
                                        },
                                    },
                                    {
                                        "name": "DYNAMICAL_SOURCE_COOP_AWS_SECRET_ACCESS_KEY",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "key": "AWS_SECRET_ACCESS_KEY",
                                                "name": "source-coop-key",
                                            }
                                        },
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
                                    {"mountPath": "/app/data", "name": "ephemeral-vol"}
                                ],
                            }
                        ],
                        "nodeSelector": {
                            "eks.amazonaws.com/compute-type": "auto",
                            "karpenter.sh/capacity-type": "spot",
                            "kubernetes.io/arch": "amd64",
                        },
                        "restartPolicy": "Never",
                        "securityContext": {
                            "fsGroup": 999,  # this is the `app` group our app runs under
                        },
                        "terminationGracePeriodSeconds": 5,
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
                            }
                        ],
                    }
                },
                "ttlSecondsAfterFinished": int(self.ttl.total_seconds()),
            },
        }


class ReformatCronJob(ReformatJob):
    name: str
    schedule: str
    ttl: timedelta = timedelta(hours=12)
    command: list[str] = ["reformat_operational_update"]

    def as_kubernetes_object(self) -> dict[str, Any]:
        job_spec = super().as_kubernetes_object()["spec"]
        return {
            "apiVersion": "batch/v1",
            "kind": "CronJob",
            "metadata": {"name": self.name},
            "spec": {
                "schedule": self.schedule,
                "concurrencyPolicy": "Replace",
                "jobTemplate": job_spec,
            },
        }
