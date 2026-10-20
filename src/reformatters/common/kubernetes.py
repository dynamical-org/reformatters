import base64
import copy
import json
import os
import random
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import pydantic
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from reformatters.common.config import Config
from reformatters.common.iterating import digest
from reformatters.common.logging import get_logger
from reformatters.common.types import Timestamp

_SECRET_MOUNT_PATH = "/secrets"  # noqa: S105
_SECRET_CONTENTS_KEY = "contents"  # noqa: S105
SERVICE_ACCOUNT = "reformatters-update-trigger"
VALIDATION_FAILURE_EXIT_CODE = 20
_CRONJOB_NAME_LABEL = "dynamical.org/cronjob-name"
_CRONJOB_UID_LABEL = "dynamical.org/cronjob-uid"
RETRY_FIRE_TIME_ANNOTATION = "dynamical.org/retry-fire-time"
RETRY_NEXT_FIRE_TIME_ANNOTATION = "dynamical.org/retry-next-fire-time"
RETRY_POLL_DEADLINE_ANNOTATION = "dynamical.org/retry-poll-deadline"
RETRY_TIMING_ANNOTATIONS = (
    RETRY_FIRE_TIME_ANNOTATION,
    RETRY_NEXT_FIRE_TIME_ANNOTATION,
    RETRY_POLL_DEADLINE_ANNOTATION,
)
RETRY_EPOCH_SECONDS_PATTERN = r"^(0|[1-9][0-9]{0,9})$"
log = get_logger(__name__)


@dataclass(frozen=True)
class RetryTiming:
    fire_time: Timestamp
    next_fire_time: Timestamp
    poll_deadline: Timestamp

    def __post_init__(self) -> None:
        for value in (self.fire_time, self.next_fire_time, self.poll_deadline):
            assert isinstance(value, pd.Timestamp), (
                "Retry timing must use pandas Timestamps"
            )
            assert not pd.isna(value), "Retry timing must contain valid timestamps"
            assert value.tzinfo is None, "Retry timing must use UTC-naive timestamps"
            assert value == value.floor("s"), "Retry timing must use whole seconds"
            assert re.fullmatch(
                RETRY_EPOCH_SECONDS_PATTERN, str(int(value.timestamp()))
            ), "Retry timing must use nonnegative epoch seconds of at most 10 digits"
        assert (
            self.fire_time
            < self.poll_deadline
            <= self.next_fire_time - timedelta(seconds=120)
        ), (
            "Retry timing must satisfy fire_time < poll_deadline <= next_fire_time - 120s"
        )

    def to_annotations(self) -> dict[str, str]:
        return {
            RETRY_FIRE_TIME_ANNOTATION: str(int(self.fire_time.timestamp())),
            RETRY_NEXT_FIRE_TIME_ANNOTATION: str(int(self.next_fire_time.timestamp())),
            RETRY_POLL_DEADLINE_ANNOTATION: str(int(self.poll_deadline.timestamp())),
        }

    @classmethod
    def from_annotations(cls, annotations: Mapping[str, str]) -> RetryTiming | None:
        present = [key in annotations for key in RETRY_TIMING_ANNOTATIONS]
        if not any(present):
            return None
        assert all(present), (
            "Retry timing annotations must be all present or all absent"
        )
        values = [annotations[key] for key in RETRY_TIMING_ANNOTATIONS]
        assert all(
            re.fullmatch(RETRY_EPOCH_SECONDS_PATTERN, value) for value in values
        ), "Retry timing annotations must be canonical nonnegative epoch seconds"
        return cls(*(pd.Timestamp(int(value), unit="s") for value in values))


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
    active_deadline: timedelta | None = None
    ttl: timedelta = timedelta(days=1)

    # Opt out of consolidation. Quick jobs have minimal impact and we'd rather not interrupt and restart longer jobs.
    pod_annotations: dict[str, str] = {"karpenter.sh/do-not-disrupt": "true"}

    secret_names: Sequence[str] = pydantic.Field(default_factory=list)
    service_account_name: str | None = None

    def mounted_secret_names(self) -> Sequence[str]:
        """Secrets mounted as JSON files at /secrets/<name>.json in the pod."""
        return self.secret_names

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
                **(
                    {"activeDeadlineSeconds": int(self.active_deadline.total_seconds())}
                    if self.active_deadline is not None
                    else {}
                ),
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
                        {
                            "action": "FailJob",
                            "onExitCodes": {
                                "containerName": "worker",
                                "operator": "In",
                                "values": [VALIDATION_FAILURE_EXIT_CODE],
                            },
                        },
                    ]
                },
                "template": {
                    "metadata": {"annotations": self.pod_annotations},
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
                                        "name": "POD_NAMESPACE",
                                        "valueFrom": {
                                            "fieldRef": {
                                                "fieldPath": "metadata.namespace"
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
                                ],
                                "image": f"{self.image}",
                                "name": "worker",
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "capabilities": {"drop": ["ALL"]},
                                },
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
                                        for secret_name in self.mounted_secret_names()
                                    ],
                                ],
                            }
                        ],
                        "nodeSelector": {
                            "eks.amazonaws.com/compute-type": "auto",
                            "karpenter.sh/capacity-type": "spot",
                        },
                        "restartPolicy": "Never",
                        **(
                            {"serviceAccountName": self.service_account_name}
                            if self.service_account_name is not None
                            else {}
                        ),
                        "securityContext": {
                            "fsGroup": 999,  # this is the `app` group our app runs under
                            "runAsNonRoot": True,
                            "runAsUser": 999,
                            "runAsGroup": 999,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "terminationGracePeriodSeconds": 30,
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
                                for secret_name in self.mounted_secret_names()
                            ],
                        ],
                    },
                },
                "ttlSecondsAfterFinished": int(self.ttl.total_seconds()),
            },
        }


# Kubernetes appends up to 11 characters when it names a cron job's jobs, so the
# cron job's own name must stay within 63 - 11.
CronJobName = Annotated[str, pydantic.Field(min_length=1, max_length=52)]


class CronJob(Job):
    name: CronJobName
    schedule: Annotated[str, pydantic.Field(min_length=1)]
    ttl: timedelta = timedelta(hours=12)
    suspend: bool = False

    def previous_fire_time(self, now: Timestamp) -> Timestamp:
        """The most recent time this schedule fired, at or before `now`.

        Every pod of one fire derives the same value, including a replacement pod
        started after an eviction, which its own start time would not give.
        """
        minutes, hours, days_of_month = _parse_schedule(self.schedule)
        fire_time = now.floor("h")
        while True:
            if fire_time.hour in hours and fire_time.day in days_of_month:
                for minute in sorted(minutes, reverse=True):
                    candidate = fire_time + timedelta(minutes=minute)
                    if candidate <= now:
                        return candidate
            fire_time -= timedelta(hours=1)

    def next_fire_time(self, now: Timestamp) -> Timestamp:
        return _next_fire_time(self.schedule, now)

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
    name: Annotated[CronJobName, pydantic.Field(pattern=r".+-update$")]
    command: Sequence[str] = ["update"]
    # Operational updates expect a single worker
    workers_total: int = 1
    parallelism: int = 1
    service_account_name: str | None = SERVICE_ACCOUNT


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


def get_deployed_cronjob_image(cronjob_name: str) -> str:
    """Read the container image of a deployed CronJob from the cluster."""
    config.load_kube_config()
    batch_v1 = client.BatchV1Api()
    cronjob = batch_v1.read_namespaced_cron_job(cronjob_name, "default")
    image = cronjob.spec.job_template.spec.template.spec.containers[0].image
    assert isinstance(image, str), f"CronJob {cronjob_name} image is not a string"
    assert len(image) > 0, f"CronJob {cronjob_name} has no container image"
    return image


def read_job_retry_timing(
    job_name: str, cron_job: CronJob, polling_duration: timedelta
) -> RetryTiming:
    config.load_incluster_config()
    job = client.BatchV1Api().read_namespaced_job(job_name, os.environ["POD_NAMESPACE"])
    annotations = job.metadata.annotations or {}
    timing = RetryTiming.from_annotations(annotations)
    if timing is not None:
        return timing
    scheduled = annotations.get("batch.kubernetes.io/cronjob-scheduled-timestamp")
    anchor = scheduled if scheduled is not None else job.metadata.creation_timestamp
    assert anchor is not None, "Root Job has no creation timestamp"
    anchor = pd.Timestamp(anchor)
    assert not pd.isna(anchor), "Root Job has an invalid timestamp"
    if anchor.tzinfo is not None:
        anchor = anchor.tz_convert("UTC").tz_localize(None)
    fire_time = anchor if scheduled is not None else cron_job.previous_fire_time(anchor)
    next_fire_time = cron_job.next_fire_time(fire_time)
    return RetryTiming(
        fire_time=fire_time,
        next_fire_time=next_fire_time,
        poll_deadline=min(
            fire_time + polling_duration, next_fire_time - timedelta(seconds=120)
        ),
    )


def create_job_from_cronjob(
    cronjob_name: str,
    job_name: str,
    *,
    skip_if_next_run_within_deadline: bool = False,
    retry_timing: RetryTiming | None = None,
    active_deadline: timedelta | None = None,
) -> bool:
    assert len(job_name) <= 63, f"Invalid Kubernetes Job name {job_name!r}"
    assert re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", job_name), (
        f"Invalid Kubernetes Job name {job_name!r}"
    )
    config.load_incluster_config()
    namespace = os.environ["POD_NAMESPACE"]
    batch_v1 = client.BatchV1Api()
    cronjob = batch_v1.read_namespaced_cron_job(cronjob_name, namespace)
    if cronjob.spec.suspend:
        log.info(f"Skipping {job_name}: CronJob {cronjob_name} is suspended")
        return False

    template = cronjob.spec.job_template
    if skip_if_next_run_within_deadline:
        deadline = template.spec.template.spec.active_deadline_seconds
        assert isinstance(deadline, int), "CronJob pod has no active deadline"
        now = pd.Timestamp(datetime.now(UTC))
        next_fire = _next_fire_time(cronjob.spec.schedule, now)
        if next_fire <= now + timedelta(seconds=deadline):
            log.info(
                f"Skipping {job_name}: CronJob {cronjob_name} next fires at {next_fire} "
                f"within its {deadline}s pod deadline"
            )
            return False

    cronjob_uid = cronjob.metadata.uid
    assert isinstance(cronjob_uid, str), "CronJob has no UID"
    labels = dict(template.metadata.labels or {}) if template.metadata else {}
    labels.update({_CRONJOB_NAME_LABEL: cronjob_name, _CRONJOB_UID_LABEL: cronjob_uid})
    annotations = dict(template.metadata.annotations or {}) if template.metadata else {}
    annotations["cronjob.kubernetes.io/instantiate"] = "manual"
    if retry_timing is not None:
        annotations.update(retry_timing.to_annotations())
    spec = copy.deepcopy(template.spec)
    if active_deadline is not None:
        assert retry_timing is not None, "A retry deadline requires retry timing"
        seconds = active_deadline.total_seconds()
        assert seconds > 0, "Retry Job deadline must be positive whole seconds"
        assert seconds.is_integer(), "Retry Job deadline must be positive whole seconds"
        assert isinstance(spec.active_deadline_seconds, int), (
            "CronJob Job template has no active deadline"
        )
        assert seconds <= spec.active_deadline_seconds, (
            "Retry Job deadline cannot exceed the template Job deadline"
        )
        spec.active_deadline_seconds = int(seconds)
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=job_name, labels=labels, annotations=annotations
        ),
        spec=spec,
    )
    try:
        batch_v1.create_namespaced_job(namespace, job)
        log.info(f"Created Job {job_name} from CronJob {cronjob_name}")
    except ApiException as error:
        if error.status != 409:
            raise
        existing = batch_v1.read_namespaced_job(job_name, namespace)
        if (
            existing.metadata.name != job_name
            or any(
                (existing.metadata.labels or {}).get(key) != value
                for key, value in (
                    (_CRONJOB_NAME_LABEL, cronjob_name),
                    (_CRONJOB_UID_LABEL, cronjob_uid),
                )
            )
            or any(
                (existing.metadata.annotations or {}).get(key) != annotations.get(key)
                for key in RETRY_TIMING_ANNOTATIONS
            )
            or existing.spec.active_deadline_seconds != spec.active_deadline_seconds
        ):
            raise
        log.info(f"Job {job_name} already exists from CronJob {cronjob_name}")
    return True


def retry_job_name(job_name: str, max_retries: int | None = 1) -> tuple[bool, str]:
    match = re.search(r"-r([1-9]\d*)$", job_name)
    retry = int(match[1]) if match else 0
    should_retry = max_retries is None or retry < max_retries
    parent = job_name[: match.start()] if match else job_name
    suffix = f"-r{retry + 1}"
    # Indexed pod hostnames append a hyphen and an int32 completion index.
    max_name_length = 52
    if len(parent) + len(suffix) <= max_name_length:
        return should_retry, parent + suffix
    parent_digest = digest([parent], length=4)
    prefix = parent[: max_name_length - len(suffix) - len(parent_digest) - 1].rstrip(
        "-"
    )
    return should_retry, f"{prefix}-{parent_digest}{suffix}"


# Operational schedules use selected minutes and hours, and either every day or
# every Nth day of the month.
_SCHEDULE_PATTERN = re.compile(
    r"(?P<minutes>\d{1,2}|\*/\d{1,2}) "
    r"(?P<hours>\*|\d{1,2}(?:,\d{1,2})*|(?:\*|\d{1,2}-\d{1,2})/\d{1,2}) "
    r"(?P<days_of_month>\*|\*/\d{1,2}) \* \*"
)


def _parse_schedule(
    schedule: str,
) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    """The minutes, hours, and days of month a cron schedule fires at."""
    match = _SCHEDULE_PATTERN.fullmatch(schedule)
    assert match is not None, (
        f"Unsupported cron schedule {schedule!r}, expected "
        "`<minutes> <hours> <days-of-month> * *` with minutes as a number or `*/N`, hours as `*`, `*/N`, "
        "`A-B/N`, or a comma separated list and days-of-month as `*` or `*/N`"
    )
    minutes = _parse_schedule_field(match["minutes"], 0, 59)
    hours = _parse_schedule_field(match["hours"], 0, 23)
    days_of_month = _parse_schedule_field(match["days_of_month"], 1, 31)
    assert minutes, f"Cron schedule {schedule!r} selects no minutes"
    assert max(minutes) <= 59, f"Cron schedule {schedule!r} has a minute above 59"
    assert hours, f"Cron schedule {schedule!r} selects no hours"
    assert max(hours) <= 23, f"Cron schedule {schedule!r} has an hour above 23"
    assert days_of_month, f"Cron schedule {schedule!r} selects no days of month"
    return minutes, hours, days_of_month


def _next_fire_time(schedule: str, now: Timestamp) -> Timestamp:
    minutes, hours, days_of_month = _parse_schedule(schedule)
    fire_time = now.floor("h")
    while True:
        if fire_time.hour in hours and fire_time.day in days_of_month:
            for minute in sorted(minutes):
                candidate = fire_time + timedelta(minutes=minute)
                if candidate > now:
                    return candidate
        fire_time += timedelta(hours=1)


def _parse_schedule_field(field: str, start: int, end: int) -> frozenset[int]:
    spec, _, step = field.partition("/")
    assert not step or int(step) > 0, "Cron schedule step must be positive"
    if "," in spec:
        return frozenset(int(value) for value in spec.split(","))
    if spec == "*":
        selected_start, selected_end = start, end
    elif "-" in spec:
        selected_start, selected_end = (int(bound) for bound in spec.split("-"))
    else:
        return frozenset({int(spec)})
    return frozenset(range(selected_start, selected_end + 1, int(step) if step else 1))
